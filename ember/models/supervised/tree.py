from ember import Tensor
from ember.models.supervised import Regression, Classification
from ember.models import track_access
from ember.datasets import Dataset
from collections import deque
from typing import Optional, Tuple, List

class RTNode(): 

  def __init__(
    self, 
    dataset: List[Tuple], 
    prediction: Optional[float] = None, 
    gain: Optional[float] = None, 
    feature: Optional[int] = None, 
    threshold: Optional[float] = None
  ): 
    self.prediction = prediction      # prediction at this current node 
    self.split_feature = feature      # which feature to split from, if any 
    self.split_threshold = threshold  # which threshold to split, if any 
    self.split_gain = gain            # amount of gain by splitting, this node
    self.left: Optional["RTNode"] = None 
    self.right: Optional["RTNode"] = None
    self.dataset = dataset            # the subdataset that is relevant to this node 

  def __repr__(self): 
    split = f"feature  = {self.split_feature} w/ "\
    f"threshold = {self.split_threshold} with gain = {self.split_gain}" \
    if self.split_feature is not None else "None"

    return f"""RTNode(
      prediction : {self.prediction} on dataset of size {len(self.dataset)}
      split : {split}
    )""" 

  def min_threshold(self, feature: int) -> Tuple[float, float, float, float]: 
    """Select the threshold that maximizes the gain on a given feature."""
    lower_ys = deque([])
    upper_ys = deque([row[1].item() for row in self.dataset]) 
  
    max_gain = 0.0
    best_threshold = 0.0
    best_pred_l = 0.0 
    best_pred_u = 0.0

    for i in range(len(self.dataset) - 1): 
      # calculate the threshold and its upper/lower predictions 
      threshold = (self.dataset[i][0][feature] + self.dataset[i + 1][0][feature]).item() / 2 
      lower_ys.append(upper_ys.popleft())
      pred_l = sum(lower_ys) / len(lower_ys)
      pred_u = sum(upper_ys) / len(upper_ys)
      pred_t = sum(lower_ys + upper_ys) / len(self.dataset)

      # calculate the new loss and gain 
      se_lower = sum([(ly - pred_l) ** 2 for ly in lower_ys])
      se_upper = sum([(uy - pred_u) ** 2 for uy in upper_ys]) 
      se_total = sum([(ty - pred_t) ** 2 for ty in lower_ys + upper_ys])
      old_mse = se_total / len(self.dataset)
      new_mse = (se_lower + se_upper) / len(self.dataset) 
      gain = old_mse - new_mse

      # if gain improves, then assign this
      if gain > max_gain: 
        max_gain = gain 
        best_threshold = threshold 
        best_pred_l = pred_l
        best_pred_u = pred_u

    return  max_gain, best_threshold, best_pred_l, best_pred_u

  def split(self) -> Tuple[int, float, float, float, float]: 
    """
    Calculates the feature and its threshold that would give maximum gain.
    This assumes that dataset.X is rank-2 tensor. Should fix this later
    """
    best_feature = -1
    best_max_gain = 0.0
    best_threshold = 0.0 
    best_pred_l = 0.0 
    best_pred_u = 0.0
    for feature in range(self.dataset[0][0].shape[0]): 
      max_gain, threshold, pred_l, pred_u = \
        self.min_threshold(feature)
      if max_gain > best_max_gain: 
        best_feature = feature
        best_max_gain = max_gain 
        best_threshold = threshold 
        best_pred_l = pred_l 
        best_pred_u = pred_u

    return best_feature, best_max_gain, best_threshold, best_pred_l, best_pred_u


class RegressionTree(Regression):  
  """Binary regression tree. """

  def __init__(self, dataset: Dataset):
    super().__init__() 
    self.dataset = dataset 

    # initialize tree to have no splits 
    self.root = RTNode(
      dataset = dataset.items(),
      prediction = dataset.Y.sum()[0].item() / len(dataset)
    ) 
    self.leaves = set([self.root])

    # compute loss for single average prediction. 
    if self.root.prediction is not None: 
      self.loss = ((dataset.Y - self.root.prediction) ** 2).sum()[0].item() / len(dataset)
    else: 
      raise Exception("Root prediction is None. Something is wrong. ")
    self.set_parameters() 

  def step(self):
    """Does a greedy split on the best leaf node with the best feature and 
    best threshold, which maximizes the gain."""

    split_node = None 
    max_gain = 0.0 
    best_feature = -1
    best_threshold = None 
    best_pred_l = None
    best_pred_u = None 

    for leaf in self.leaves: # for each leaf node that we could split on 
      # calculate the feature and its threshold that would give maximum gain 
      feature, gain, threshold, pred_u, pred_l = leaf.split()

      if gain > max_gain: 
        max_gain = gain
        split_node = leaf 
        best_feature = feature 
        best_threshold = threshold 
        best_pred_u = pred_u 
        best_pred_l = pred_l 

    if split_node is not None:
      # finally take the node that has max gain and split it 
      split_node.split_feature = best_feature       # type: ignore
      split_node.split_threshold = best_threshold   # type: ignore
      split_node.split_gain = max_gain              # type: ignore

      # now split the dataset at the split node according to 
      # the optimal feature and threshold 

      ds = sorted(
        split_node.dataset,
        key = lambda x : x[0][best_feature].item()
      )

      ds_left = [r for r in ds if r[0][best_feature].item() < best_threshold]
      ds_right = [r for r in ds if r[0][best_feature].item() >= best_threshold]
      left_node = RTNode(ds_left, best_pred_l)
      right_node = RTNode(ds_right, best_pred_u)

      split_node.left = left_node     
      split_node.right = right_node

      # update leaf node set
      self.leaves.remove(split_node) 
      self.leaves.add(left_node)
      self.leaves.add(right_node)

  @track_access
  def forward(self, X: Tensor) -> Tensor: 
    super().forward(X)
    raise NotImplementedError()

class ClassificationTree(Classification): 

  def __init__(self):
    super().__init__() 
    self.set_parameters()

  @track_access
  def forward(self, X: Tensor) -> Tensor: 
    super().forward(X)
    raise NotImplementedError()

