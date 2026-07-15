use ash::vk;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::vulkan::VkBackend;
use super::triple_buffer::{TripleBuffer, BufferSlotState};
use super::StreamTelemetry;

pub struct TransferEngine {
    backend: Arc<VkBackend>,
    pub triple_buffer: TripleBuffer,

    transfer_pool: vk::CommandPool,
    transfer_cmd: vk::CommandBuffer,
    transfer_fence: vk::Fence,

    timeline_sem: vk::Semaphore,
    timeline_val: AtomicU64,

    staging_buf: vk::Buffer,
    staging_mem: vk::DeviceMemory,
    staging_mapped: *mut u8,
    staging_size: u64,

    pub telemetry: StreamTelemetry,
}

unsafe impl Send for TransferEngine {}
unsafe impl Sync for TransferEngine {}

impl TransferEngine {
    pub fn new(backend: &Arc<VkBackend>, slot_size: u64) -> Result<Self, crate::vulkan::VkError> {
        let triple_buffer = TripleBuffer::new(backend, slot_size)?;

        let transfer_pool = unsafe {
            backend.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(backend._queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };

        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(transfer_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cbs = unsafe { backend.device.allocate_command_buffers(&cmd_alloc)? };
        let transfer_cmd = cbs[0];

        let transfer_fence = unsafe {
            backend.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };

        let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem_info = vk::SemaphoreCreateInfo::default()
            .push_next(&mut timeline_type);
        let timeline_sem = unsafe { backend.device.create_semaphore(&sem_info, None)? };

        let staging_size = slot_size;
        let (staging_buf, staging_mem) = backend.allocate_buffer(
            staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let staging_mapped = unsafe {
            backend.device.map_memory(staging_mem, 0, staging_size, vk::MemoryMapFlags::empty())?
        } as *mut u8;

        Ok(Self {
            backend: backend.clone(),
            triple_buffer,
            transfer_pool,
            transfer_cmd,
            transfer_fence,
            timeline_sem,
            timeline_val: AtomicU64::new(0),
            staging_buf,
            staging_mem,
            staging_mapped,
            staging_size,
            telemetry: StreamTelemetry::new(),
        })
    }

    pub fn current_timeline(&self) -> u64 {
        self.timeline_val.load(Ordering::Acquire)
    }

    fn next_timeline(&self) -> u64 {
        self.timeline_val.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn upload_slot(&mut self, data: &[u8]) -> Option<(usize, u64)> {
        let slot_size = self.triple_buffer.max_slot_size as usize;
        if data.len() > slot_size {
            return None;
        }

        let (idx, _) = self.triple_buffer.acquire_write_slot()?;
        let dst = &self.triple_buffer.slots[idx];
        let timeline_val = self.next_timeline();

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.staging_mapped,
                data.len(),
            );
        }

        let region = vk::BufferCopy::default()
            .size(data.len() as u64)
            .src_offset(0)
            .dst_offset(0);

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.backend.device.reset_command_buffer(
                self.transfer_cmd,
                vk::CommandBufferResetFlags::empty(),
            ).ok()?;
            self.backend.device.begin_command_buffer(self.transfer_cmd, &begin).ok()?;
            self.backend.device.cmd_copy_buffer(
                self.transfer_cmd,
                self.staging_buf,
                dst.buffer,
                &[region],
            );
            self.backend.device.end_command_buffer(self.transfer_cmd).ok()?;
        }

        let signal_values = [timeline_val];
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&signal_values);

        let cbs = [self.transfer_cmd];
        let submit = vk::SubmitInfo::default()
            .command_buffers(&cbs)
            .push_next(&mut timeline_info);

        let submits = [submit];
        unsafe {
            self.backend.device.reset_fences(&[self.transfer_fence]).ok()?;
            self.backend.device.queue_submit(
                self.backend._queue,
                &submits,
                self.transfer_fence,
            ).ok()?;
        }

        self.triple_buffer.mark_device_ready(idx, timeline_val);
        self.telemetry.record_transfer(data.len() as u64);

        Some((idx, timeline_val))
    }

    pub fn wait_for_timeline(&self, value: u64, timeout_ns: u64) -> bool {
        let sem = [self.timeline_sem];
        let vals = [value];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&sem)
            .values(&vals);
        unsafe {
            self.backend.device.wait_semaphores(&wait, timeout_ns).is_ok()
        }
    }

    pub fn get_timeline_value_host(&self) -> Result<u64, crate::vulkan::VkError> {
        let value = unsafe {
            self.backend.device.get_semaphore_counter_value(self.timeline_sem)?
        };
        Ok(value)
    }

    pub fn sync_all(&self) {
        let val = self.current_timeline();
        if val > 0 {
            self.wait_for_timeline(val, u64::MAX);
        }
    }

    pub fn staging_mut(&self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.staging_mapped, self.staging_size as usize)
        }
    }
}

impl Drop for TransferEngine {
    fn drop(&mut self) {
        self.sync_all();
        unsafe {
            if self.transfer_fence != vk::Fence::null() {
                self.backend.device.destroy_fence(self.transfer_fence, None);
            }
            if self.transfer_pool != vk::CommandPool::null() {
                self.backend.device.destroy_command_pool(self.transfer_pool, None);
            }
            if self.timeline_sem != vk::Semaphore::null() {
                self.backend.device.destroy_semaphore(self.timeline_sem, None);
            }
            if self.staging_buf != vk::Buffer::null() {
                self.backend.device.destroy_buffer(self.staging_buf, None);
            }
            if self.staging_mem != vk::DeviceMemory::null() {
                self.backend.device.free_memory(self.staging_mem, None);
            }
        }
    }
}
