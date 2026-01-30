from reward_model_support import RewardModel

###########################
# Define model families
###########################

class Llama3Model(RewardModel):
    pass

class Gemma2Model(RewardModel):
    pass


###########################
# Define individual models
###########################

class nicolinho__QRM_Gemma_2_27B(Gemma2Model):
    def _extract_scores_from_outputs(self, outputs):
        return outputs.score.cpu().float().squeeze(-1).tolist()
RewardModel.MODEL_REGISTRY['nicolinho/QRM-Gemma-2-27B'] = nicolinho__QRM_Gemma_2_27B

class Skywork__Skywork_Reward_Gemma_2_27B(Gemma2Model):
    def _extract_scores_from_outputs(self, outputs):
        return outputs.logits.squeeze(-1).cpu().tolist()
RewardModel.MODEL_REGISTRY['Skywork/Skywork-Reward-Gemma-2-27B-v0.2'] = Skywork__Skywork_Reward_Gemma_2_27B
RewardModel.MODEL_REGISTRY['Skywork/Skywork-Reward-Gemma-2-27B'] = Skywork__Skywork_Reward_Gemma_2_27B

class Skywork__Skywork_Reward_Llama_3_1_8B_v0_2(Llama3Model):
    def _extract_scores_from_outputs(self, outputs):
        return outputs.logits.squeeze(-1).cpu().tolist()
RewardModel.MODEL_REGISTRY['Skywork/Skywork-Reward-Llama-3.1-8B-v0.2'] = Skywork__Skywork_Reward_Llama_3_1_8B_v0_2

class Nicolinho__QRM_Llama3_1_8B(Llama3Model):
    def _extract_scores_from_outputs(self, outputs):
        return outputs.score.cpu().float().squeeze(-1).tolist()
RewardModel.MODEL_REGISTRY['nicolinho/QRM-Llama3.1-8B'] = Nicolinho__QRM_Llama3_1_8B

class LxzGordon__URM_LlaMa_3_1_8B(Llama3Model):
    def _extract_scores_from_outputs(self, outputs):
        return outputs.logits[:, 0].tolist()
RewardModel.MODEL_REGISTRY['LxzGordon/URM-LLaMa-3.1-8B'] = LxzGordon__URM_LlaMa_3_1_8B

class Ray2333__GRM_llama3_8B_rewardmodel_ft(Llama3Model):
    def _extract_scores_from_outputs(self, outputs):
        return outputs.logits.squeeze(-1).cpu().tolist()
RewardModel.MODEL_REGISTRY['Ray2333/GRM-Llama3-8B-rewardmodel-ft'] = Ray2333__GRM_llama3_8B_rewardmodel_ft

class Ray2333__GRM_Llama3_2_3B_rewardmodel_ft(Llama3Model):
    def _extract_scores_from_outputs(self, outputs):
        return outputs.logits.squeeze(-1).cpu().tolist()
RewardModel.MODEL_REGISTRY['Ray2333/GRM-Llama3.2-3B-rewardmodel-ft'] = Ray2333__GRM_Llama3_2_3B_rewardmodel_ft

class RLHFlow__ArmoRM_Llama3_8B_v0_1(Llama3Model):
    def _extract_scores_from_outputs(self, outputs):
        return outputs.score.cpu().float().tolist()
RewardModel.MODEL_REGISTRY['RLHFlow/ArmoRM-Llama3-8B-v0.1'] = RLHFlow__ArmoRM_Llama3_8B_v0_1

class Ray2333__GRM_gemma2_2B_rewardmodel_ft(Gemma2Model):
    def _extract_scores_from_outputs(self, outputs):
        return outputs.logits.squeeze(-1).cpu().tolist()
RewardModel.MODEL_REGISTRY['Ray2333/GRM-gemma2-2B-rewardmodel-ft'] = Ray2333__GRM_gemma2_2B_rewardmodel_ft
