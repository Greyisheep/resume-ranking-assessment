def mean_reciprocal_rank(ranked_cvs: list, ground_truth: list) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) for ranked CVs.

    Args:
        ranked_cvs (list): A list of dictionaries containing filenames and their relevance scores.
        ground_truth (list): A list of filenames in the correct relevance order.

    Returns:
        float: The MRR score.
    """
    ranks = []
    for item in ranked_cvs:
        if item['filename'] in ground_truth:
            rank = ground_truth.index(item['filename']) + 1
            ranks.append(1 / rank)

    return sum(ranks) / len(ground_truth)

def evaluate_and_log_mrr(ranked_cvs: list, ground_truth: list, logger) -> None:
    """
    Evaluate the MRR and log the result.

    Args:
        ranked_cvs (list): A list of dictionaries containing filenames and their relevance scores.
        ground_truth (list): A list of filenames in the correct relevance order.
        logger: Logger object for logging the MRR result.
    """
    mrr_score = mean_reciprocal_rank(ranked_cvs, ground_truth)
    logger.info(f"MRR Score: {mrr_score:.4f}")
