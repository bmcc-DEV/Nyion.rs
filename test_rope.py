import math

n_dims = 64
for i in range(n_dims // 2):
    # What llama.cpp does? Let's check exactly how LLaMA defines it.
    # LLaMA paper says: Theta_i = 10000 ^ (-2i/d) for i = 0..d/2-1
    # So exponents are 0/d, -2/d, -4/d...
    print(f"k={i}, LLaMA paper: {-2*i/n_dims}")
