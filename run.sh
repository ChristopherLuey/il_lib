#!/bin/bash
# il_lib policy server launcher for Franka eval/correction
# Starts serve.py with the selected task + policy checkpoint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source conda
if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [[ -f ~/miniforge3/etc/profile.d/conda.sh ]]; then
    source ~/miniforge3/etc/profile.d/conda.sh
fi
conda activate behavior

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ====== Checkpoint paths (best val loss for each) ======
CKPT_BASE="$SCRIPT_DIR/outputs/2026-05-13/03-48-59/dp_franka_mug_n075_20260513-034859/ckpt/step_190000-epoch_1021-loss_0.0018.pth"
CKPT_HG_R1="$SCRIPT_DIR/outputs/2026-05-13/23-38-06/dp_franka_mug_n075_corr1_hgdagger_20260513-233806/ckpt/step_110000-epoch_407-loss_0.0192.pth"
CKPT_HG_R2="$SCRIPT_DIR/outputs/2026-05-14/21-36-59/dp_franka_mug_n075_corr2_hgdagger_20260514-213659/ckpt/step_110000-epoch_311-loss_0.0222.pth"
CKPT_HG_R3="$SCRIPT_DIR/outputs/2026-05-15/19-47-09/dp_franka_mug_n075_corr3_hgdagger_20260515-194709/ckpt/step_150000-epoch_340-loss_0.0193.pth"
CKPT_YALCIN_R1="$SCRIPT_DIR/outputs/2026-05-26/17-03-35/dp_franka_mug_n075_corr1_yalcin_hgdagger_20260526-170335/ckpt/step_200000-epoch_757-loss_0.0329.pth"
CKPT_YALCIN_R12="$SCRIPT_DIR/outputs/2026-05-27/01-18-37/diffusion_rgb_franka_mug_20260527-011837/ckpt/step_170000-epoch_484-loss_0.0409.pth"
CKPT_YALCIN_R25="$SCRIPT_DIR/outputs/2026-05-27/17-21-54/diffusion_rgb_franka_mug_20260527-172154/ckpt/step_140000-epoch_532-loss_0.0187.pth"
CKPT_STEF_R1="$SCRIPT_DIR/outputs/2026-05-26/19-19-20/diffusion_rgb_franka_mug_20260526-191920/ckpt/step_130000-epoch_460-loss_0.0189.pth"
CKPT_STEF_R25="$SCRIPT_DIR/outputs/2026-05-27/17-32-21/diffusion_rgb_franka_mug_20260527-173221/ckpt/step_180000-epoch_664-loss_0.0257.pth"
CKPT_SIRIUS_R1=""
CKPT_SIRIUS_R2=""
CKPT_SIRIUS_R3=""
CKPT_HANG_BASE="$SCRIPT_DIR/outputs/2026-08-22/00-19-42/dp_franka_hang_50_20260822-001942/ckpt/step_200000-epoch_1176-loss_0.0019.pth"

# ====== Task selection ======
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     IL-Lib Policy Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Select task:"
echo "  1) mug/base"
echo "  2) real/base (hang)"
echo ""
read -p "Task [1]: " task_choice
task_choice=${task_choice:-1}

case $task_choice in
    1|mug) TASK="mug/base" ;;
    2|hang|real) TASK="real/base" ;;
    *) echo -e "${RED}Unknown task${NC}"; exit 1 ;;
esac

# ====== Policy selection ======
echo ""
echo "Select policy:"
echo ""
echo -e "  ${BLUE}--- HG-DAgger ---${NC}"
echo "  1) Base (75 demos, val_loss=0.0018)"
echo "  2) HG-DAgger R1 (75+25, val_loss=0.0192)"
echo "  3) HG-DAgger R2 (75+50, val_loss=0.0222)"
echo "  4) HG-DAgger R3 (75+75, val_loss=0.0193)"
echo ""
echo -e "  ${BLUE}--- Yalcin ---${NC}"
echo "  5) Yalcin R1 (75+25, val_loss=0.0329)"
echo "  6) Yalcin R1+R2 (75+50, val_loss=0.0409)"
echo "  7) Yalcin 25 corr (75+25, val_loss=0.0187)"
echo ""
echo -e "  ${BLUE}--- Stef ---${NC}"
echo "  8) Stef R1 (75+25, val_loss=0.0189)"
echo "  9) Stef 25 corr (75+25, val_loss=0.0257)"
echo ""
echo -e "  ${BLUE}--- Sirius ---${NC}"
echo "  10) Sirius R1 (not trained)"
echo "  11) Sirius R2 (not trained)"
echo "  12) Sirius R3 (not trained)"
echo ""
echo -e "  ${BLUE}--- Hang ---${NC}"
echo "  13) Hang Base (50 demos, loss=0.0019; use task 2)"
echo ""
read -p "Policy [1]: " policy_choice
policy_choice=${policy_choice:-1}

case $policy_choice in
    1|base)      CKPT="$CKPT_BASE";        POLICY_NAME="base" ;;
    2|hg1)       CKPT="$CKPT_HG_R1";       POLICY_NAME="hgdagger_r1" ;;
    3|hg2)       CKPT="$CKPT_HG_R2";       POLICY_NAME="hgdagger_r2" ;;
    4|hg3)       CKPT="$CKPT_HG_R3";       POLICY_NAME="hgdagger_r3" ;;
    5|y1)        CKPT="$CKPT_YALCIN_R1";   POLICY_NAME="yalcin_r1" ;;
    6|y12)       CKPT="$CKPT_YALCIN_R12";  POLICY_NAME="yalcin_r12" ;;
    7|y25)       CKPT="$CKPT_YALCIN_R25";  POLICY_NAME="yalcin_25corr" ;;
    8|st1)       CKPT="$CKPT_STEF_R1";     POLICY_NAME="stef_r1" ;;
    9|st25)      CKPT="$CKPT_STEF_R25";    POLICY_NAME="stef_25corr" ;;
    10|s1)       CKPT="$CKPT_SIRIUS_R1";   POLICY_NAME="sirius_r1" ;;
    11|s2)       CKPT="$CKPT_SIRIUS_R2";   POLICY_NAME="sirius_r2" ;;
    12|s3)       CKPT="$CKPT_SIRIUS_R3";   POLICY_NAME="sirius_r3" ;;
    13|hang)     CKPT="$CKPT_HANG_BASE";   POLICY_NAME="hang_base" ;;
    *)           echo -e "${RED}Unknown policy${NC}"; exit 1 ;;
esac

if [[ -z "$CKPT" ]]; then
    echo -e "${RED}ERROR: No checkpoint available for $POLICY_NAME (not yet trained)${NC}"
    exit 1
fi

if [[ ! -f "$CKPT" ]]; then
    echo -e "${RED}ERROR: Checkpoint not found: $CKPT${NC}"
    exit 1
fi

# ====== Launch ======
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "  Task:   ${BLUE}$TASK${NC}"
echo -e "  Policy: ${BLUE}$POLICY_NAME${NC}"
echo -e "  Ckpt:   ${YELLOW}$(basename $CKPT)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Starting policy server on port 8000...${NC}"
echo ""

exec python serve.py arch=dp_franka robot=franka_iiil task=$TASK ckpt_path="$CKPT"
