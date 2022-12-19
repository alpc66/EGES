import pandas as pd
import numpy as np
from itertools import chain
import pickle
import time
import networkx as nx
from walker import RandomWalker
from sklearn.preprocessing import LabelEncoder
import argparse

def get_graph_context_all_pairs(walks, window_size):
  all_pairs = []
  for k in range(len(walks)):
    for i in range(len(walks[k])):
      for j in range(i - window_size, i + window_size + 1):
        if i == j or j < 0 or j >= len(walks[k]):
          continue
        else:
          all_pairs.append([walks[k][i], walks[k][j]])
  return np.array(all_pairs, dtype=np.int32)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='manual to this script')
  parser.add_argument("--data_path", type=str, default='./data/')
  parser.add_argument("--p", type=float, default=0.25)
  parser.add_argument("--q", type=float, default=2)
  parser.add_argument("--num_walks", type=int, default=10)
  parser.add_argument("--walk_length", type=int, default=10)
  parser.add_argument("--window_size", type=int, default=5)
  args = parser.parse_known_args()[0]

  action_data = pd.read_csv(args.data_path + 'graph.csv').dropna()
  all_skus = action_data['entity_id'].unique()
  all_skus = pd.DataFrame({'entity_id': list(all_skus)})
  sku_lbe = LabelEncoder()
  all_skus['entity_id'] = sku_lbe.fit_transform(all_skus['entity_id'])
  action_data['entity_id'] = sku_lbe.transform(action_data['entity_id'])

  # session2graph
  G = nx.read_edgelist('./data/graph.csv',
                       create_using=nx.DiGraph(),
                       nodetype=None,
                       data=[('weight', int)])
  walker = RandomWalker(G, p=args.p, q=args.q)
  print("Preprocess transition probs...")
  walker.preprocess_transition_probs()

  session_reproduce = walker.simulate_walks(num_walks=args.num_walks,
                                            walk_length=args.walk_length,
                                            workers=16,
                                            verbose=1)
  session_reproduce = list(filter(lambda x: len(x) > 2, session_reproduce))

  # add side info
  product_data = pd.read_csv(args.data_path + 'features.csv').dropna()

  all_skus['entity_id'] = sku_lbe.inverse_transform(all_skus['entity_id'])
  print("sku nums: " + str(all_skus.count()))
  sku_side_info = pd.merge(all_skus, product_data, on='sku_id',
                           how='left').fillna(0)

  # id2index
  for feat in sku_side_info.columns:
    if feat != 'sku_id':
      lbe = LabelEncoder()
      sku_side_info[feat] = lbe.fit_transform(sku_side_info[feat])
    elif feat in ['actor1', 'actor2', 'actor3', 'actor4', 'genres1', 'genres2']:
      sku_side_info[feat] = sku_lbe.transform(sku_side_info[feat])

  sku_side_info = sku_side_info.sort_values(by=['entity_id'], ascending=True)
  sku_side_info.to_csv('./data_cache/sku_side_info.csv',
                       index=False,
                       header=False,
                       sep='\t')

  # get pair
  all_pairs = get_graph_context_all_pairs(session_reproduce, args.window_size)
  np.savetxt('./data_cache/all_pairs', X=all_pairs, fmt="%d", delimiter=" ")
