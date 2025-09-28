from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import matthews_corrcoef as mcc
from scipy.stats import pearsonr


@dataclass
class BaseTask:
    def default_loss(self, input, output, targets, **kwargs):
        return F.binary_cross_entropy(output, targets)
        # return F.l1_loss(output, targets)

    def process_for_loss(self, **kwargs):
        return kwargs


class PBValidityTask(BaseTask):
    def default_loss(self, input, output, targets, **kwargs):
        return F.binary_cross_entropy(output, targets)

    def mae_loss(self, input, output, targets, **kwargs):
        return F.l1_loss(targets, output)

    def matthews_correlation(self, input, output, targets, **kwargs):
        output = (output > 0.5).float()
        return mcc(output.cpu(), targets.cpu())


class PBClassifiyTask(BaseTask):
    def default_loss(self, input, output, targets, **kwargs):
        return F.binary_cross_entropy_with_logits(output, targets)

    def mae_loss(self, input, output, targets, **kwargs):
        return F.l1_loss(nn.Sigmoid()(output), targets)

    def matthews_correlation(self, input, output, targets, **kwargs):
        output = nn.Sigmoid()(output)
        output = (output > 0.5).float()
        return mcc(output.cpu(), targets.cpu())
    
    def mse_loss(self, input, output, targets, **kwargs):
        return F.mse_loss(nn.Sigmoid()(output), targets)


class InteractionSimilarity(BaseTask):
    def default_loss(self, input, output, targets, **kwargs):
        return F.mse_loss(output, targets)

    def mae_loss(self, input, output, targets, **kwargs):
        return F.l1_loss(targets, output)

    def pearson_r(self, input, output, targets, **kwargs):
        return pearsonr(output.detach().cpu(), targets.detach().cpu())[0]


task_registry = {
    "base": BaseTask,
    "pbvalidity": PBValidityTask,
    "pbclassify": PBClassifiyTask,
    "interaction_similarity": InteractionSimilarity,
}
