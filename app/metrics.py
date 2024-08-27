import math
from bert_score import score as bert_score

def mean_reciprocal_rank(ranked_cvs: list, weighted_scores: list) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) for a list of ranked CVs and their corresponding weighted scores.

    The MRR is a statistic measure for evaluating any process that produces a list of possible responses to a sample of queries, ordered by probability of correctness.

    Args:
        ranked_cvs (list): A list of ranked CVs.
        weighted_scores (list): A list of weighted scores corresponding to the ranked CVs.

    Returns:
        float: The Mean Reciprocal Rank (MRR) value.
    """
    ranks = []
    for idx, item in enumerate(ranked_cvs):
        if weighted_scores[idx] > 0:
            ranks.append(1 / (idx + 1))
    return sum(ranks) / len(ranked_cvs) if ranked_cvs else 0.0
def evaluate_and_log_mrr(ranked_cvs: list, logger) -> None:
    """
    Evaluates the Mean Reciprocal Rank (MRR) score for a list of ranked CVs and logs the result.

    Args:
        ranked_cvs (list): A list of dictionaries where each dictionary represents a CV and contains a 'score' key.
        logger: A logging object used to log the MRR score.

    Returns:
        None
    """
    # Apply weighted scoring
    weighted_scores = [cv['score'] for cv in ranked_cvs]
    mrr_score = mean_reciprocal_rank(ranked_cvs, weighted_scores)
    logger.info(f"MRR Score: {mrr_score:.4f}")
def dcg(scores):
    return sum(score / math.log2(idx + 2) for idx, score in enumerate(scores))

def ndcg(ranked_cvs, logger):
    relevance_scores = [cv['score'] for cv in ranked_cvs]
    actual_dcg = dcg(relevance_scores)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg(ideal_relevance_scores)
    ndcg_score = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    logger.info(f"NDCG Score: {ndcg_score:.4f}")

def evaluate_and_log_ndcg(ranked_cvs, logger):
    """
    Evaluates the Normalized Discounted Cumulative Gain (NDCG) for the given ranked CVs and logs the result.

    Args:
        ranked_cvs (list): A list of ranked CVs to evaluate.
        logger (Logger): A logger instance to log the NDCG result.
    """
    ndcg(ranked_cvs, logger)
def log_bert_scores(generated_summary, logger):
    """
    Logs the BERT scores (Precision, Recall, and F1) for a given generated summary.

    Args:
        generated_summary (str): The summary text generated by the model.
        logger (logging.Logger): The logger instance used to log the BERT scores.

    Returns:
        None
    """
    P, R, F1 = bert_score(generated_summary, generated_summary, lang="en")
    logger.info(f"BERTScore - Precision: {P.mean().item():.4f}")
    logger.info(f"BERTScore - Recall: {R.mean().item():.4f}")
    logger.info(f"BERTScore - F1: {F1.mean().item():.4f}")
