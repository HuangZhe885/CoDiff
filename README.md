# CoDiff: Conditional Diffusion Model for Collaborative 3D Object Detection

1. This work builds upon the previously published RoCo paper, attempting to use diffusion models to address noise issues (pose errors and time delays) in collaborative perception. It was conducted in July 2024.  

![image](https://github.com/user-attachments/assets/ec088148-7217-4110-88a8-02a75906da60)  

2. Documenting the training process to avoid forgetting:  
[Training Process Documentation](https://lx2xygwjrgr.feishu.cn/docx/AgrcdAIuOoIR9zxdozUcpXPZndh)  

3. The main modified files are:  
   - `point_pillar_baseline_multiscale.py`  
   - `diffusion_fuse.py` (which implements using single-vehicle features as conditions to guide the diffusion process for generating ensemble features)  

4. Due to varying feature map size requirements across datasets, the code is quite messy... only I can understand it.  

5. Nevertheless, the results are promising - even outperforming RoCo. However, diffusion-generated features perform poorly in high-noise scenarios, likely because the model learns from suboptimal samples, leading to inferior outputs.  

![image](https://github.com/user-attachments/assets/83ed8904-7f36-4914-9eb0-c1dedfbdc6c4)

