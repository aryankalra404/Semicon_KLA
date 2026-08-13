# References

The solution and presentation use the following public sources.

1. Chen et al., “Simple Baselines for Image Restoration,” ECCV 2022. The
   activation-free restoration blocks motivate the compact NAF-style backbone.
   <https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3043_ECCV_2022_paper.php>
2. Wang et al., “Real-ESRGAN: Training Real-World Blind Super-Resolution with
   Pure Synthetic Data,” ICCV Workshops 2021. This motivates compound
   degradation modeling and order diversity.
   <https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html>
3. Zhang et al., “Designing a Practical Degradation Model for Deep Blind Image
   Super-Resolution,” ICCV 2021. This motivates randomized blur, downsampling,
   and noise operations for blind restoration robustness.
   <https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Designing_a_Practical_Degradation_Model_for_Deep_Blind_Image_Super-Resolution_ICCV_2021_paper.html>
4. Wang et al., “Image Quality Assessment: From Error Visibility to Structural
   Similarity,” IEEE Transactions on Image Processing, 2004. Source for SSIM.
   <https://doi.org/10.1109/TIP.2003.819861>
5. Zhang et al., “The Unreasonable Effectiveness of Deep Features as a
   Perceptual Metric,” CVPR 2018. Source for LPIPS.
   <https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html>
6. Barron, “A General and Adaptive Robust Loss Function,” CVPR 2019. Related
   background for robust pixel-space objectives.
   <https://openaccess.thecvf.com/content_CVPR_2019/html/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.html>
7. SEMICON India Hackathon 2026, KLA “AI-Based Restoration of Degraded Images”
   official problem and submission requirements.
   <https://i4c.in/hackathon-2026/>

The official training and blind-test arrays are provided by KLA for this
hackathon and are not redistributed beyond the required restored predictions.
