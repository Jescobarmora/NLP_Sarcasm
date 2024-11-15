from sklearn.metrics import roc_auc_score, accuracy_score

class Evaluator:
    def evaluate(self, y_true, y_pred, y_probs):
        roc_auc = roc_auc_score(y_true, y_probs)
        accuracy = accuracy_score(y_true, y_pred)
        return roc_auc, accuracy