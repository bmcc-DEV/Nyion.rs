use ash::vk;
use std::sync::Arc;

use crate::vulkan::VkBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferSlotState {
    Free,
    Staging,
    DeviceReady,
    Computing,
}

pub struct BufferSlot {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub state: BufferSlotState,
    pub timeline_value: u64,
}

impl BufferSlot {
    pub fn new(b: vk::Buffer, m: vk::DeviceMemory, size: u64) -> Self {
        Self {
            buffer: b,
            memory: m,
            size,
            state: BufferSlotState::Free,
            timeline_value: 0,
        }
    }
}

pub struct TripleBuffer {
    pub slots: [BufferSlot; 3],
    write_index: usize,
    pub max_slot_size: u64,
}

impl TripleBuffer {
    pub fn new(backend: &Arc<VkBackend>, slot_size: u64) -> Result<Self, crate::vulkan::VkError> {
        let usage = vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let mem_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        let slots = [
            Self::alloc_slot(backend, slot_size, usage, mem_flags)?,
            Self::alloc_slot(backend, slot_size, usage, mem_flags)?,
            Self::alloc_slot(backend, slot_size, usage, mem_flags)?,
        ];

        Ok(Self { slots, write_index: 0, max_slot_size: slot_size })
    }

    fn alloc_slot(
        backend: &Arc<VkBackend>,
        size: u64,
        usage: vk::BufferUsageFlags,
        mem_flags: vk::MemoryPropertyFlags,
    ) -> Result<BufferSlot, crate::vulkan::VkError> {
        let (buffer, memory) = backend.allocate_buffer(size, usage, mem_flags)?;
        Ok(BufferSlot::new(buffer, memory, size))
    }

    pub fn acquire_write_slot(&mut self) -> Option<(usize, &BufferSlot)> {
        for offset in 0..3 {
            let idx = (self.write_index + offset) % 3;
            if self.slots[idx].state == BufferSlotState::Free {
                self.slots[idx].state = BufferSlotState::Staging;
                self.write_index = idx;
                return Some((idx, &self.slots[idx]));
            }
        }
        None
    }

    pub fn mark_device_ready(&mut self, idx: usize, timeline_value: u64) {
        if idx < 3 {
            self.slots[idx].state = BufferSlotState::DeviceReady;
            self.slots[idx].timeline_value = timeline_value;
        }
    }

    pub fn acquire_compute_slot(&mut self) -> Option<(usize, &BufferSlot)> {
        for i in 0..3 {
            if self.slots[i].state == BufferSlotState::DeviceReady {
                self.slots[i].state = BufferSlotState::Computing;
                return Some((i, &self.slots[i]));
            }
        }
        None
    }

    pub fn release_slot(&mut self, idx: usize) {
        if idx < 3 {
            self.slots[idx].state = BufferSlotState::Free;
        }
    }

    pub fn slot_count_ready(&self) -> usize {
        self.slots.iter().filter(|s| s.state == BufferSlotState::DeviceReady).count()
    }

    pub fn slot_count_free(&self) -> usize {
        self.slots.iter().filter(|s| s.state == BufferSlotState::Free).count()
    }
}

impl Drop for TripleBuffer {
    fn drop(&mut self) {
    }
}
