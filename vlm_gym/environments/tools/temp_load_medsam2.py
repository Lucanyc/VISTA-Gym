import sys
import os
import torch

sys.path.append("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/MedSAM2")
os.chdir("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/MedSAM2")

from sam2.build_sam import build_sam2_video_predictor_npz

try:
    predictor = build_sam2_video_predictor_npz(
        "sam2/configs/sam2.1_hiera_t512.yaml",
        "checkpoints/MedSAM2_latest.pt",
        device="cuda",
        mode="eval",
        apply_postprocessing=True
    )

    torch.save(predictor, "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/temp_medsam2_predictor.pt")
    print("Model saved successfully")

except Exception as e:
    print(f"Error in subprocess: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)