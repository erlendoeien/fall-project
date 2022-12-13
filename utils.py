import logging
from transformers4rec.torch.ranking_metric import NDCGAt, AvgPrecisionAt, RecallAt, PrecisionAt#, MeanReciprocalRankAt


def setup_logging(log_file, **kwargs):
    config = {**{"filename":log_file,
            "format":'%(asctime)s | %(levelname)s: %(message)s',
            "level": logging.DEBUG, 
            "filemode":"w"}, **kwargs}
    logging.basicConfig(**config)
    
    
# Defines the evaluation top-N metrics and the cut-offs
METRICS = [NDCGAt(top_ks=[5, 10, 20, 50], labels_onehot=True),  
           RecallAt(top_ks=[5, 10, 20, 50], labels_onehot=True),
           PrecisionAt(top_ks=[5, 10, 20, 50], labels_onehot=True),
           AvgPrecisionAt(top_ks=[5, 10, 20, 50], labels_onehot=True),
          ]
