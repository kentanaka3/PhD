import os
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
IMG_PATH = os.path.join(PRJ_PATH, "img")
DATA_PATH = os.path.join(PRJ_PATH, "data")
import sys
# Add to path
if INC_PATH not in sys.path: sys.path.append(INC_PATH)
import copy
import argparse
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta as td
from obspy.core.utcdatetime import UTCDateTime
from sklearn.metrics import ConfusionMatrixDisplay as ConfMtxDisp

from constants import *
import initializer as ini
import parser as prs

THRESHOLDS : list[float] = [round(t, 2) for t in np.linspace(0.1, 0.9, 9)]
DATES = None

def plot_data(TRUE : pd.DataFrame, PRED : pd.DataFrame,
              args : argparse.Namespace, phase = PWAVE) -> None:
  """
  input  :
    - TRUE          (pd.DataFrame)
    - PRED          (pd.DataFrame)
    - args          (argparse.Namespace)
    - phase         (str)

  output :
    - None

  errors :
    - AttributeError

  notes  :
    The data is plotted for each model and weight. The plots are saved in the
    img directory.
  """
  global DATES
  MSG = f"Cumulative number of {phase} picks"
  if args.verbose: print(MSG)
  start, end = args.dates
  PRED = PRED[(PRED[PHASE_STR] == phase) &
              (PRED[TIMESTAMP_STR] >= start.datetime)]
  TRUE = TRUE[(TRUE[PHASE_STR] == phase)].reset_index(drop=True)
  if DATES is None:
    DATES = [start.datetime]
    while DATES[-1] <= end.datetime: DATES.append(DATES[-1] + ONE_DAY)

  y_true = [len(TRUE[TRUE[TIMESTAMP_STR] <= d].index) for d in DATES]
  # Plot a 2x2 grid for each model and weight
  for model, dtfrm in PRED.groupby(MODEL_STR):
    _, _axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = _axs.flatten()
    plt.suptitle(model, fontsize=16)
    axs[0].set(xticklabels=[], xlabel=None, ylabel=MSG)
    axs[1].set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None)
    axs[2].set(xlabel="Date", ylabel=MSG)
    axs[3].set(xlabel="Date", yticklabels=[], ylabel=None)
    y_max = 0
    for i, (weight, data) in enumerate(dtfrm.groupby(WEIGHT_STR)):
      axs[i].set_title(weight)
      for threshold in THRESHOLDS:
        y = [len(data[(data[PROBABILITY_STR] >= threshold) &
                      (data[TIMESTAMP_STR] <= d)].index) for d in DATES]
        axs[i].plot([np.datetime64(t) for t in DATES], y,
                    label=rf"$\geq$ {threshold}")
        y_max = max(y_max, max(y))
      axs[i].plot([np.datetime64(t) for t in DATES], y_true, label="True",
                  color="k")
      y_max = max(y_max, max(y_true))
    for ax in axs:
      ax.set(xlim=(DATES[0], DATES[-1]), ylim=(1, y_max), yscale="log")
      ax.grid()
      ax.legend()
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in axs[2].get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in axs[3].get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    IMG_FILE = \
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join([CMTV_PICKS_STR, model, phase]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()
    if args.verbose: print(f"Saving {IMG_FILE}")
  # Plot a 2x2 grid for each model, network and station
  for (model, network, station), dtfrm in \
    PRED.groupby([MODEL_STR, NETWORK_STR, STATION_STR]):
    _, _axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = _axs.flatten()
    plt.suptitle(SPACE_STR.join([model, network, station]), fontsize=16)
    axs[0].set(xticklabels=[], xlabel=None, ylabel=MSG)
    axs[1].set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None)
    axs[2].set(xlabel="Date", ylabel=MSG)
    axs[3].set(xlabel="Date", yticklabels=[], ylabel=None)
    y_max = 0
    for i, (weight, data) in enumerate(dtfrm.groupby(WEIGHT_STR)):
      axs[i].set_title(weight)
      for threshold in THRESHOLDS:
        y = [len(data[(data[PROBABILITY_STR] >= threshold) &
                      (data[TIMESTAMP_STR] <= d)].index) for d in DATES]
        axs[i].plot([np.datetime64(t) for t in DATES], y,
                    label=rf"$\geq$ {threshold}")
        y_max = max(y_max, max(y))
    for ax in axs:
      ax.set_ylim(0, y_max)
      ax.grid()
      ax.legend()
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in axs[2].get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in axs[3].get_xticklabels():
      label.set(rotation=30, horizontalalignment='right')
    IMG_FILE = \
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join([CMTV_PICKS_STR, model, network,
                                station, phase]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()
    if args.verbose: print(f"Saving {IMG_FILE}")

def dist_balanced(T : pd.Series, P : pd.Series) -> float:
  return (dist_time(T, P) + 9. * dist_phase(T, P)) / 10.

def dist_phase(T : pd.Series, P : pd.Series) -> float:
  return int(P[PHASE_STR] == T[PHASE_STR])

def dist_time(T : pd.Series, P : pd.Series,
              offset : td = PICK_OFFSET) -> float:
  return 1. - (diff_time(T, P) / offset)

def diff_time(T : pd.Series, P : pd.Series) -> float:
  return td(seconds=abs(P[TIMESTAMP_STR] - T[TIMESTAMP_STR]))

def dist_default(T : pd.Series, P : pd.Series) -> float:
  return (99. * dist_balanced(T, P) + P[PROBABILITY_STR]) / 100.

def plot_timeline(G : nx.Graph, pos : dict[int, tuple[float, int]], N : int,
                  model_name : str, dataset_name : str) -> None:
  fig, ax = plt.subplots(figsize=(15, 2))
  node_color = [COLOR_ENCODING[G.nodes[node][STATUS_STR]]\
                  [G.nodes[node][PHASE_STR]] for node in G.nodes]
  nx.draw(G, pos=pos, ax=ax, node_color=node_color, edge_color='black',
          width=2, node_size=10)
  ax.axis('on')
  ax.tick_params(bottom=True, labelbottom=True)
  for node in G.nodes:
    if node < N:
      x = pos[node][0]
      ax.axvline(x=x + PICK_OFFSET.microseconds * 1e-6, c='k', ls='--')
      ax.axvline(x=x - PICK_OFFSET.microseconds * 1e-6, c='k', ls='--')
  ax.grid()
  ax.set_title(SPACE_STR.join([model_name, dataset_name]))
  fig.tight_layout()
  plt.show()
  plt.close(fig=fig)

# remove entry based on best_pred_idx (#1 clean phase)
def remove_entry_with_pred_idx(tp_list, best_pred_idx):
    tp_list[:] = [entry for entry in tp_list if entry[8] != best_pred_idx]  # entry[8] corresponds to best_pred_idx
# remove entry based on best_true_idx (#2 clean phase)
def remove_entry_with_true_idx(tp_list, best_true_idx):
    tp_list[:] = [entry for entry in tp_list if entry[7] != best_true_idx]  # entry[7] corresponds to best_true_idx
# remove entry in FP
def remove_in_fp_fn(fp_list, timestamp):
    fp_list[:] = [entry for entry in fp_list if entry[4] != timestamp] # entry 4 is the timestamp of the pick to remove
# check element not in both lists
def find_unique(list1, list2):
    return [item for item in list1 if item not in list2]

def conf_mtx2(TRUE : pd.DataFrame, PRED : pd.DataFrame, model_name : str,
              dataset_name : str, args : argparse.Namespace) -> pd.DataFrame:
  N = len(TRUE.index)
  M = len(PRED.index)
  start, _ = args.dates
  tags = [PWAVE, SWAVE, NONE_STR]
  CFN_MTX = pd.DataFrame(0, index=tags, columns=tags, dtype=int)
  TRUE[STATUS_STR] = FN_STR
  PRED[STATUS_STR] = FP_STR
  print(TRUE)
  print(PRED)
  # backup of indices
  TRUE['index'] = TRUE.index
  PRED['index'] = PRED.index
  # Lists to store, day by day, True Positives, False Positives, False Negatives, phases mismatches for the 3 loops
  # these lists are cleared at next iteration
  # 1st loop: predicted P over True
  TP_PredP_part, FP_PredP_part = [], []
  # true S, predicted P cases
  tS_pP_PredP_part = []
  # 2nd loop: predicted S over true
  TP_PredS_part, FP_PredS_part= [], []
  # true P, predicted S cases
  tP_pS_PredS_part = []
  # 3rd loop: true (P and S) over all predicted
  TP_TrueP_part, FN_TrueP_part = [], []
  TP_TrueS_part, FN_TrueS_part = [], []
  # true P, predicted S cases
  tP_pS_TrueP_part = []
  # true S, predicted P cases
  tS_pP_TrueS_part = []
  # Set to track predicted picks
  matched_indices_pred = []
  # 1st Loop over predicted P-phase picks
  p_pred_picks_day = PRED[PRED[PHASE_STR] == PWAVE].reset_index(drop=True) # Only predicted P-phase picks
  for j, pred_pick in p_pred_picks_day.iterrows():
    # Compare with all true picks (regardless of phase)
    # If the time difference is within PICK_OFFSET, check for the closest match
    time_diffs = abs(TRUE['timestamp'] - pred_pick['timestamp'])
    mask = time_diffs <= PICK_OFFSET

    if not time_diffs[mask].empty:
      min_time_diff = time_diffs[mask].min()
      best_true_idx = time_diffs[mask].idxmin()
      best_pred_idx = p_pred_picks_day['index'].iloc[j] # referred to the PRED dataframe

      # Case of True Positive or False Positive for P-phase
      best_true_pick = TRUE.loc[best_true_idx]
      matched_indices_pred.append([best_true_idx, best_pred_idx,best_true_pick[PHASE_STR],pred_pick[PHASE_STR],min_time_diff,pred_pick['probability']]) #keep track of matched indices

      if best_true_pick[PHASE_STR] == PWAVE:
        # True Positive: P predicted and true phase match
        TP_PredP_part.append([pred_pick[PHASE_STR], min_time_diff, 
                              best_true_pick['quality'], best_true_pick[PHASE_STR],best_true_idx,best_pred_idx,best_true_pick['timestamp'],pred_pick['timestamp']])
      elif best_true_pick[PHASE_STR] == SWAVE:
        # true S and predicted P phases differ
        tS_pP_PredP_part.append([pred_pick[PHASE_STR], min_time_diff, 
                                 best_true_pick['quality'], best_true_pick[PHASE_STR],best_true_idx,best_pred_idx,best_true_pick['timestamp'],pred_pick['timestamp']])
    else:
      # False Positive: no true pick matches within the time window
      FP_PredP_part.append([pred_pick[PHASE_STR], pred_pick['timestamp'], pred_pick['probability']])
    # close loop

  # 2nd Loop over predicted S picks for true picks
  s_pred_picks_day = PRED[PRED[PHASE_STR] == SWAVE].reset_index(drop=True)

  for j, pred_pick in s_pred_picks_day.iterrows():
    # Compare with all true picks (regardless of phase)
    time_diffs = abs(TRUE['timestamp'] - pred_pick['timestamp'])
    mask = time_diffs <= PICK_OFFSET

    if not time_diffs[mask].empty:
      min_time_diff = time_diffs[mask].min()
      best_true_idx = time_diffs[mask].idxmin()
      best_pred_idx = s_pred_picks_day['index'].iloc[j] # referred to the PRED dataframe

      # Case of True Positive or False Positive
      best_true_pick = TRUE.loc[best_true_idx]
      matched_indices_pred.append([best_true_idx, best_pred_idx,best_true_pick[PHASE_STR],pred_pick[PHASE_STR],min_time_diff,pred_pick['probability']]) # keep track of matched indices
      
      if best_true_pick[PHASE_STR] == SWAVE:
        # True Positive: pred and true phases coincide and it's S
        TP_PredS_part.append([pred_pick[PHASE_STR], min_time_diff,
                            best_true_pick['quality'], best_true_pick[PHASE_STR],best_true_idx,best_pred_idx,best_true_pick['timestamp'],pred_pick['timestamp'] ])
      elif best_true_pick[PHASE_STR] == PWAVE:
        # true P, predicted S case
        tP_pS_PredS_part.append([pred_pick[PHASE_STR], min_time_diff,
                            best_true_pick['quality'], best_true_pick[PHASE_STR],best_true_idx,best_pred_idx,best_true_pick['timestamp'], pred_pick['timestamp']])
    else:
      # False Positive: no true pick matches within the window
      FP_PredS_part.append([pred_pick[PHASE_STR], pred_pick['timestamp'], pred_pick['probability']])
    # close loop

  # Set to track matched true picks
  matched_indices_true = []

  # 3rd loop over true picks
  for i, true_pick in TRUE.iterrows():                  
    if true_pick[PHASE_STR] == PWAVE:  # True P picks
      time_diffs = abs(true_pick['timestamp'] - PRED['timestamp'])
      # If the time difference is within PICK_OFFSET, check for the closest match
      mask = time_diffs <= PICK_OFFSET
      if not time_diffs[mask].empty:
        best_true_idx = i
        #min_time_diff = time_diffs[mask].min()
        #best_pred_idx = time_diffs[mask].idxmin()
        # min time diff with max probability
        filtered_df = PRED[mask].copy()
        filtered_df['time_difference'] = time_diffs[mask].values
        min_time_entry = filtered_df.loc[filtered_df.groupby(
            'time_difference')['probability'].idxmax()].nsmallest(1, 'time_difference')
        min_time_diff = min_time_entry['time_difference'].iloc[0]
        best_pred_idx = min_time_entry['index'].iloc[0]
        # check for presence of minimum difference in the window
        best_pred_pick = PRED.loc[best_pred_idx]                             
        matched_indices_true.append([best_true_idx, best_pred_idx,true_pick[PHASE_STR],best_pred_pick[PHASE_STR],min_time_diff,best_pred_pick['probability']])
        # Compare with predicted P picks
        if best_pred_pick[PHASE_STR] == PWAVE:
          # True Positive: P pick
          TP_TrueP_part.append([best_pred_pick[PHASE_STR], min_time_diff, 
                              true_pick['quality'], true_pick[PHASE_STR],best_true_idx,best_pred_idx,true_pick['timestamp'],best_pred_pick['timestamp']])
        elif best_pred_pick[PHASE_STR] == SWAVE:
          # true P, predicted S case
          tP_pS_TrueP_part.append([best_pred_pick[PHASE_STR], min_time_diff, 
                              true_pick['quality'], true_pick[PHASE_STR],best_true_idx,best_pred_idx,true_pick['timestamp'],best_pred_pick['timestamp']])
      else:
        # no presence of predicted in the window, False Negative case
        FN_TrueP_part.append([true_pick[PHASE_STR], true_pick['timestamp'], true_pick['quality']])
    elif true_pick[PHASE_STR] == SWAVE:  # True S picks
      # If the time difference is within PICK_OFFSET, check for the closest match
      time_diffs = abs(true_pick['timestamp'] - PRED['timestamp'])
      # If the time difference is within PICK_OFFSET, check for the closest match
      mask = time_diffs <= PICK_OFFSET
      if not time_diffs[mask].empty:
        best_true_idx = i
        #min_time_diff = time_diffs[mask].min()
        #best_pred_idx = time_diffs[mask].idxmin()
        filtered_df = PRED[mask].copy()
        filtered_df['time_difference'] = time_diffs[mask]
        min_time_entry = filtered_df.loc[filtered_df.groupby(
            'time_difference')['probability'].idxmax()].nsmallest(1, 'time_difference')
        min_time_diff = min_time_entry['time_difference'].iloc[0]
        best_pred_idx = min_time_entry['index'].iloc[0]
        
        # check for presence of minimum difference in the window
        best_pred_pick = PRED.loc[best_pred_idx]                             
        matched_indices_true.append([best_true_idx, best_pred_idx,true_pick[PHASE_STR],best_pred_pick[PHASE_STR],min_time_diff,best_pred_pick['probability']])
        # Predicted S picks                        
        if best_pred_pick[PHASE_STR] == SWAVE:  
          # True Positive: S pick
          TP_TrueS_part.append([best_pred_pick[PHASE_STR], min_time_diff, 
                              true_pick['quality'], true_pick[PHASE_STR],best_true_idx,best_pred_idx,true_pick['timestamp'],best_pred_pick['timestamp']])
        elif best_pred_pick[PHASE_STR] == PWAVE:
          # true S, predicted P case
          tS_pP_TrueS_part.append([best_pred_pick[PHASE_STR], min_time_diff, 
                              true_pick['quality'], true_pick[PHASE_STR],best_true_idx,best_pred_idx,true_pick['timestamp'],best_pred_pick['timestamp']])
      else:
        # no presence of predicted in the window, False Negative case
        FN_TrueS_part.append([true_pick[PHASE_STR], true_pick['timestamp'], true_pick['quality']])
      # close loop

  #print(matched_indices_pred)
  #print(matched_indices_true)

  # CASE: PICKS WITH SAME TIME DIFFERENCE, BUT DIFFERENT PROBABILITIES
  final_updated_true = matched_indices_true.copy()
  final_updated_pred = matched_indices_pred.copy()

  for true_row in final_updated_true:
    true_best_idx = true_row[0]
    true_time_diff = true_row[4]

    for pred_row in final_updated_pred:
      pred_best_idx = pred_row[0]
      pred_time_diff = pred_row[4]

      # Check for correspondance
      if true_best_idx == pred_best_idx and true_time_diff == pred_time_diff:
        # Probabilities comparison
        if true_row[5] > pred_row[5]:  # true_row has better probability -> substitute predicted with true
          print(f"P(pred) < P(true): P({pred_best_idx},{pred_row[1]}) < P({true_best_idx},{true_row[1]})")
          final_updated_pred.remove(pred_row)  # Remove pred_row
          final_updated_pred.append(true_row)  # Add true_row
          to_remove = [PRED.loc[pred_row[1]][PHASE_STR],
                                PRED.loc[pred_row[1]]['timestamp'],
                                PRED.loc[pred_row[1]]['probability']]
          to_append = [PRED.loc[true_row[1]][PHASE_STR], true_time_diff, 
                       TRUE.loc[true_best_idx]['quality'],
                       TRUE.loc[true_best_idx][PHASE_STR], true_best_idx,
                       true_row[1], TRUE.loc[true_best_idx]['timestamp'],
                       PRED.loc[true_row[1]]['timestamp']]
          # remove predicted and add to FP
          if pred_row[2] == pred_row[3] == PWAVE:
            remove_entry_with_pred_idx(TP_PredP_part, pred_row[1])
            FP_PredP_part.append(to_remove)
          elif pred_row[2] == pred_row[3] == SWAVE:
            remove_entry_with_pred_idx(TP_PredS_part, pred_row[1])
            FP_PredS_part.append(to_remove)
          elif pred_row[2] == PWAVE and pred_row[3] == SWAVE:
            remove_entry_with_pred_idx(tP_pS_PredS_part, pred_row[1])
            FP_PredS_part.append(to_remove)
          elif pred_row[2] == SWAVE and pred_row[3] == PWAVE:
            remove_entry_with_pred_idx(tS_pP_PredP_part, pred_row[1])
            FP_PredP_part.append(to_remove)
          # now add the true, IF NOT ALREADY IN THE PRED LIST
          if true_row[2] == true_row[3] == PWAVE and not true_row in final_updated_pred:
            # add true to TP_PredP
            TP_PredP_part.append(to_append)
          elif true_row[2] == true_row[3] == SWAVE and not true_row in final_updated_pred:
            # add true to TP_PredS
            TP_PredS_part.append(to_append)
          # discard phase mismatch entry
          elif true_row[2] == PWAVE and true_row[3] == SWAVE and not true_row in final_updated_pred:
            # add true to tP_pS_TrueP
            tP_pS_PredS_part.append(to_append)
          elif true_row[2] == SWAVE and true_row[3] == PWAVE and not true_row in final_updated_pred:
            # add true to tS_pP_TrueP
            tS_pP_PredP_part.append(to_append)                                    
        elif pred_row[5] > true_row[5]:  # pred_row has better probability -> substitute true with pred    
          print(f"P pred > P true: P({pred_best_idx},{pred_row[1]}) > P({true_best_idx},{true_row[1]})")
          final_updated_true.remove(true_row)  # Remove true_row                                 
          final_updated_true.append(pred_row)  # Add pred_row
          to_remove = [TRUE.loc[true_best_idx][PHASE_STR],
                       TRUE.loc[true_best_idx]['timestamp'],
                       TRUE.loc[true_best_idx]['quality']]
          to_append = [PRED.loc[pred_row[1]][PHASE_STR], pred_time_diff,
                       TRUE.loc[pred_best_idx]['quality'],
                       TRUE.loc[pred_best_idx][PHASE_STR], pred_best_idx,
                       pred_row[1], TRUE.loc[pred_best_idx]['timestamp'],
                       PRED.loc[pred_row[1]]['timestamp']]
          # remove true and add to FN
          if true_row[2] == true_row[3] == PWAVE:
            # add to False Negatives
            remove_entry_with_true_idx(TP_TrueP_part, true_best_idx)
            FN_TrueP_part.append(to_remove)
          elif true_row[2] == true_row[3] == SWAVE:
            # add to False Negatives
            remove_entry_with_true_idx(TP_TrueS_part, true_best_idx)
            FN_TrueS_part.append(to_remove)
          elif true_row[2] == PWAVE and true_row[3] == SWAVE:
            remove_entry_with_true_idx(tP_pS_TrueP_part, true_best_idx)
            FN_TrueP_part.append(to_remove) 
          elif true_row[2] == SWAVE and true_row[3] == PWAVE:
            remove_entry_with_true_idx(tS_pP_TrueS_part, true_best_idx)
            FN_TrueS_part.append(to_remove)
          # then add the predicted IF NOT ALREADY IN THE TRUE LIST
          if pred_row[2] == pred_row[3] == PWAVE and not pred_row in final_updated_true:
            # add pred to TP_TrueP
            TP_TrueP_part.append(to_append)
          elif pred_row[2] == pred_row[3] == SWAVE and not pred_row in final_updated_true:
            # add pred to TP_TrueS
            TP_TrueS_part.append(to_append)
          # discard phase mismatch entry
          elif pred_row[2] == PWAVE and pred_row[3] == SWAVE and not pred_row in final_updated_true:
            # add pred to tP_pS_TrueP 
            tP_pS_TrueP_part.append(to_append)
          elif true_row[2] == SWAVE and true_row[3] == PWAVE and not pred_row in final_updated_true:                                  
            # add pred to tS_pP_TrueS
            tS_pP_TrueS_part.append(to_append)

  #final check
  clean_pred = []
  clean_true = []
  # 1st CLEAN PHASE AFTER LOOP
  # find the smallest time difference for each best_pred_idx with same best_true_idx
  min_diff_dict_pred = {}
  for t in final_updated_pred:
    best_true_idx, best_pred_idx, phase_true, phase_pred, min_time_diff, prob_pred = t
    if (best_true_idx not in min_diff_dict_pred) or (min_time_diff < min_diff_dict_pred[best_true_idx][4]):
      min_diff_dict_pred[best_true_idx] = t
  
  # handle duplicates
  for t in final_updated_pred:
    best_true_idx, best_pred_idx, phase_true, phase_pred, min_time_diff, prob_pred = t
    if t == min_diff_dict_pred[best_true_idx]:
      clean_pred.append([best_true_idx, best_pred_idx, phase_true, phase_pred, min_time_diff, prob_pred])
    else:
      to_fp = [PRED.loc[best_pred_idx][PHASE_STR],
               PRED.loc[best_pred_idx]['timestamp'],
               PRED.loc[best_pred_idx]['probability']]
      # discard same phase entry
      if phase_true == phase_pred and phase_true == PWAVE:
        remove_entry_with_pred_idx(TP_PredP_part, best_pred_idx)
        # add to False Positives
        FP_PredP_part.append(to_fp)
      elif phase_true == phase_pred and phase_true == SWAVE:
        remove_entry_with_pred_idx(TP_PredS_part, best_pred_idx)
        # add to False Positives
        FP_PredS_part.append(to_fp)
      # discard phase mismatch entry
      elif phase_true == PWAVE and phase_pred == SWAVE:
        remove_entry_with_pred_idx(tP_pS_PredS_part, best_pred_idx)
        FP_PredS_part.append(to_fp)
      elif phase_true == SWAVE and phase_pred == PWAVE:
        remove_entry_with_pred_idx(tS_pP_PredP_part, best_pred_idx)
        FP_PredP_part.append(to_fp) 
  
  # 2nd CLEAN PHASE AFTER LOOP
  # find the smallest time difference for each best_true_idx with same best_pred_idx
  min_diff_dict_true = {}
  for t in final_updated_true:
    best_true_idx, best_pred_idx, phase_true, phase_pred, min_time_diff, prob_pred = t
    if (best_pred_idx not in min_diff_dict_true) or (min_time_diff < min_diff_dict_true[best_pred_idx][4]):
      min_diff_dict_true[best_pred_idx] = t

  # handle duplicates
  for t in final_updated_true:
    best_true_idx, best_pred_idx, phase_true, phase_pred, min_time_diff, prob_pred = t
    if t == min_diff_dict_true[best_pred_idx]:
      clean_true.append([best_true_idx, best_pred_idx, phase_true, phase_pred, min_time_diff, prob_pred])
    else:
      to_fn = [TRUE.loc[best_true_idx][PHASE_STR],
               TRUE.loc[best_true_idx]['timestamp'],
               TRUE.loc[best_true_idx]['quality']]
      # discard same phase entry
      if phase_true == phase_pred == PWAVE:
        remove_entry_with_true_idx(TP_TrueP_part, best_true_idx)
        # add to False Negatives
        FN_TrueP_part.append(to_fn)
      elif phase_true == phase_pred == SWAVE:
        remove_entry_with_true_idx(TP_TrueS_part, best_true_idx)
        # add to False Negatives
        FN_TrueS_part.append(to_fn)
      # discard phase mismatch entry
      elif phase_true == PWAVE and phase_pred == SWAVE:
        remove_entry_with_true_idx(tP_pS_TrueP_part, best_true_idx)
        FN_TrueP_part.append(to_fn)
      elif phase_true == SWAVE and phase_pred == PWAVE:
        remove_entry_with_true_idx(tS_pP_TrueS_part, best_true_idx)
        FN_TrueS_part.append(to_fn)

  # case: a pred pick was wrongly associated and put among false positives, 
  # but it is still present in clean_true
  missing_pred = find_unique(clean_true,clean_pred)
  missing_true = find_unique(clean_pred,clean_true)
  #print(missing_pred)
  #print(missing_true)
  if missing_pred:
    for pick in missing_pred:
      to_add = [pick[3], pick[4], TRUE.loc[pick[0]]['quality'], pick[2],
                pick[0], pick[1], TRUE.loc[pick[0]]['timestamp'],
                PRED.loc[pick[1]]['timestamp']]
      if pick[2] == pick[3] == PWAVE:
        TP_PredP_part.append(to_add)
        # remove from FP if present
        remove_in_fp_fn(FP_PredP_part, PRED.loc[pick[1]]['timestamp'])
      if pick[2] == pick[3] == SWAVE:
        TP_PredS_part.append(to_add)
        # remove from FP if present
        remove_in_fp_fn(FP_PredS_part, PRED.loc[pick[1]]['timestamp'])
      if pick[2] == PWAVE and pick[3] == SWAVE:
        tP_pS_PredS_part.append(to_add)
        # remove from FP if present
        remove_in_fp_fn(FP_PredS_part, PRED.loc[pick[1]]['timestamp'])
      if pick[2] == SWAVE and pick[3] == PWAVE:
        tS_pP_PredP_part.append(to_add)
        # remove from FP if present
        remove_in_fp_fn(FP_PredP_part, PRED.loc[pick[1]]['timestamp'])
  elif missing_true:
    for pick in missing_true:
      to_add = [pick[3], pick[4], TRUE.loc[pick[0]]['quality'], pick[2],
                pick[0],pick[1],TRUE.loc[pick[0]]['timestamp'],
                PRED.loc[pick[1]]['timestamp']]
      if pick[2] == pick[3] == PWAVE:
        TP_TrueP_part.append(to_add)
        # remove from FN if present
        remove_in_fp_fn(FN_TrueP_part, TRUE.loc[pick[0]]['timestamp'])
      if pick[2] == pick[3] == SWAVE:
        TP_TrueS_part.append(to_add)
        # remove from FN if present
        remove_in_fp_fn(FN_TrueS_part, TRUE.loc[pick[0]]['timestamp'])
      if pick[2] == PWAVE and pick[3] == SWAVE:
        tP_pS_TrueP_part.append(to_add)
        # remove from FN if present
        remove_in_fp_fn(FN_TrueP_part, TRUE.loc[pick[0]]['timestamp'])
      if pick[2] == SWAVE and pick[3] == PWAVE:
        tS_pP_PredP_part.append(to_add)
        # remove from FP if present
        remove_in_fp_fn(FN_TrueS_part, TRUE.loc[pick[0]]['timestamp'])
  exit()

def conf_mtx(TRUE : pd.DataFrame, PRED : pd.DataFrame, model_name : str,
             dataset_name : str, threshold : float, args : argparse.Namespace)\
      -> list[pd.DataFrame, list, list, list]:
  """
  input  :
    - TRUE          (pd.DataFrame)
    - PRED          (pd.DataFrame)
    - model_name    (str)
    - dataset_name  (str)
    - threshold     (float)
    - args          (argparse.Namespace)

  output :
    - pd.DataFrame
    - list
    - list
    - list

  errors :
    - AttributeError

  notes  :

  """
  N = len(TRUE.index)
  M = len(PRED.index)
  G = nx.Graph()
  # All Predictions are initialized as False Positives
  G.add_nodes_from([(i + N, {PHASE_STR : P[PHASE_STR], STATUS_STR : FP_STR})
                    for i, P in PRED.iterrows()], bipartite=1)
  start, _ = args.dates
  # Position of the nodes in the graph for plotting
  pos = {i + N : (P[TIMESTAMP_STR] - start, 1) for i, P in PRED.iterrows()}
  for i, T in TRUE.iterrows():
    # All True are initialized as False Negatives
    G.add_nodes_from([(i, {PHASE_STR : T[PHASE_STR], STATUS_STR : FN_STR})],
                     bipartite=0)
    # Position of the nodes in the graph for plotting
    pos[i] = (T[TIMESTAMP_STR] - start, 0)
    # Temporal difference between the True and Predicted picks
    # TODO: Consider H71 error interval
    # PICKS = PRED[PRED[TEMPORAL_STR] < H71_OFFSET[T[WEIGHT_STR]]]
    PICKS = PRED[(PRED[TIMESTAMP_STR] - T[TIMESTAMP_STR])\
                 .apply(lambda x : td(seconds=abs(x))) <= PICK_OFFSET]
    if PICKS.empty: continue
    # If there are picks within the OFFSET, we change the status of the True
    # and Predicted picks to True Positive and we add the corresponding edges
    # to the graph with the weight of the edge being the distance between the
    # True and Predicted picks
    G.nodes[i][STATUS_STR] = TP_STR
    for j, P in PICKS.iterrows():
      G.add_edge(i, j + N, weight=dist_default(T, P))
      G.nodes[j + N][STATUS_STR] = TP_STR
  LINKS = nx.max_weight_matching(G)
  # As there are more Predicted picks than True picks, we only traverse the
  # True picks of the graph and remove the edges that are not part of the
  # maximum weight matching
  for node in range(N):
    for neighbor in copy.deepcopy(G.neighbors(node)):
      #               TRUE, PRED            PRED, TRUE
      edge, edge_r = (node, neighbor), (neighbor, node)
      if not (edge in LINKS or edge_r in LINKS): G.remove_edge(*edge)
  TP, FN, FP = set(), [], set()
  tags = [PWAVE, SWAVE, NONE_STR]
  CFN_MTX = pd.DataFrame(0, index=tags, columns=tags, dtype=int)
  # We traverse the TRUE nodes of the graph to update the status and append
  # the relevant information to the True Positives and False Negatives lists
  for node in range(N):
    # As we have removed the edges that are not part of the maximum weight
    # matching, we can now assign the status of the nodes that are not part of
    # the matching graph
    t = TRUE.iloc[node]
    if nx.degree(G, node):
      p = PRED.iloc[list(G.neighbors(node))[0] - N]
      CFN_MTX.loc[t[PHASE_STR], p[PHASE_STR]] += 1
      if t[PHASE_STR] == p[PHASE_STR]:
        TP.add((model_name, dataset_name, None, t[ID_STR],
                (str(t[TIMESTAMP_STR]), str(p[TIMESTAMP_STR])),
                (t[PROBABILITY_STR], p[PROBABILITY_STR]), t[PHASE_STR],
                p[NETWORK_STR], t[STATION_STR]))
    else:
      G.nodes[node][STATUS_STR] = FN_STR
      FN.append([model_name, dataset_name, threshold, t[ID_STR],
                 t[TIMESTAMP_STR].__str__(), t[PROBABILITY_STR], t[PHASE_STR],
                 None, t[STATION_STR]])
      CFN_MTX.loc[t[PHASE_STR], NONE_STR] += 1

  # We traverse the PRED nodes of the graph to update the status and append
  # the relevant information to the False Positives list
  for node in range(N, N + M):
    if not nx.degree(G, node):
      G.nodes[node][STATUS_STR] = FP_STR
      p = PRED.iloc[node - N]
      FP.add((model_name, dataset_name, None, None, str(p[TIMESTAMP_STR]),
              p[PROBABILITY_STR], p[PHASE_STR], p[NETWORK_STR],
              p[STATION_STR]))
      CFN_MTX.loc[NONE_STR, p[PHASE_STR]] += 1
  if args.interactive: plot_timeline(G, pos, N, model_name, dataset_name)
  return CFN_MTX, TP, FN, FP

def stat_test(TRUE : pd.DataFrame, PRED : pd.DataFrame,
              args : argparse.Namespace, method = "Picker") -> pd.DataFrame:
  """
  input  :
    - TRUE          (pd.DataFrame)
    - PRED          (pd.DataFrame)
    - args          (argparse.Namespace)

  output :
    - pd.DataFrame

  errors :
    - AttributeError

  notes  :
    | MODEL | WEIGHT | STATION | THRESHOLD | PHASE | TIMESTAMP | TYPE |
    -------------------------------------------------------------------
  """
  if args.verbose: print("Computing the Confusion Matrix")
  start, end = args.dates
  N_seconds = int((end - start) / (2 * PICK_OFFSET.total_seconds()))
  TP, FN, FP = set(), [], set()
  for threshold, (model, dataframe_m) in \
    itertools.product(THRESHOLDS, PRED.groupby(MODEL_STR)):
    fig, _axs = plt.subplots(2, 2, figsize=(10, 9))
    axs = _axs.flatten()
    plt.rcParams.update({'font.size': 12})
    for ax, (weight, dataframe_w) in zip(axs, dataframe_m.groupby(WEIGHT_STR)):
      ax.set_title(weight)
      tags = [PWAVE, SWAVE, NONE_STR]
      CFN_MTX = pd.DataFrame(0, index=tags, columns=tags, dtype=int)
      for station, dataframe_s in dataframe_w.groupby(STATION_STR):
        TRUE_S = TRUE[TRUE[STATION_STR] == station].reset_index(drop=True)
        PRED_S = dataframe_s[dataframe_s[PROBABILITY_STR] >= threshold]\
                  .reset_index(drop=True)
        cfn_mtx, tp, fn, fp = conf_mtx(TRUE_S, PRED_S, model, weight,
                                       threshold, args)
        CFN_MTX += cfn_mtx
        TP = TP.union(tp)
        FN.extend(fn)
        FP = FP.union(fp)
      CFN_MTX.loc[NONE_STR, NONE_STR] = N_seconds - CFN_MTX.sum().sum()
      disp = ConfMtxDisp(CFN_MTX.values, display_labels=CFN_MTX.columns)
      disp.plot(ax=ax, colorbar=False)
      for labels in disp.text_.ravel():
        labels.set(color="#E4007C", fontsize=12, fontweight="bold")
      disp.im_.set(clim=(1, N_seconds), cmap="Blues", norm="log")
    axs[0].set()
    axs[1].set(ylabel=None, yticklabels=[])
    axs[2].set(title=None)
    axs[2].set_xlabel(args.weights[2], fontsize=14)
    axs[2].xaxis.tick_top()
    axs[3].set(ylabel=None, yticklabels=[], title=None)
    axs[3].set_xlabel(args.weights[3], fontsize=14)
    axs[3].xaxis.tick_top()
    fig.subplots_adjust(left=0.08, right=1.08, top=.95, bottom=0.05,
                        wspace=0.1, hspace=0.2)
    fig.colorbar(disp.im_, ax=axs, orientation='vertical',
                 label="Number of Picks", aspect=50, shrink=0.8)
    disp.im_.set_clim(1, N_seconds)
    IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                    UNDERSCORE_STR.join([method, CFN_MTX_STR, model,
                                         str(threshold)]) + PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()
  # True Positives
  TP = pd.DataFrame(TP, columns=HEADER_PRED).sort_values(SORT_HIERARCHY_PRED)
  TP_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                  UNDERSCORE_STR.join([method, TP_STR]) + CSV_EXT)
  TP.to_csv(TP_FILE, index=False)
  # False Negatives
  FN = pd.DataFrame(FN, columns=HEADER_PRED).sort_values(SORT_HIERARCHY_PRED)
  FN_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                 UNDERSCORE_STR.join([method, FN_STR]) + CSV_EXT)
  FN.to_csv(FN_FILE, index=False)
  # False Negative Pie plot
  for (model, weight, phase, threshold), dtfrm in \
    FN.groupby([MODEL_STR, WEIGHT_STR, PHASE_STR, THRESHOLD_STR]):
    fig, ax = plt.subplots(figsize=(5, 5))
    dtfrm[PROBABILITY_STR].value_counts().plot(kind='pie', ax=ax,
                                                   autopct='%1.1f%%')
    IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                    UNDERSCORE_STR.join([method, FN_STR, model, weight, phase,
                                         str(threshold)]) + PNG_EXT)
    plt.savefig(IMG_FILE)
    plt.close()
  # False Positives
  FP = pd.DataFrame(FP, columns=HEADER_PRED).sort_values(SORT_HIERARCHY_PRED)
  FP_FILE = Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                 UNDERSCORE_STR.join([method, FP_STR]) + CSV_EXT)
  FP.to_csv(FP_FILE, index=False)
  return TP
  # TODO: Redo the plots for the True Positives, False Negatives and False
  #       Positives
  # Plot the True Positives, False Negatives histogram and the Recall as a
  # function of the threshold for each model and weight
  groups = [MODEL_STR, WEIGHT_STR, PHASE_STR]
  m = max(TP.groupby(groups)[THRESHOLD_STR].value_counts().max(),
          FN.groupby(groups)[THRESHOLD_STR].value_counts().max())
  m = (m + 9) // 10 * 10
  for model in args.models:
    _, _axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = _axs.flatten()
    for ax1, weight in zip(axs, args.weights):
      ax2 = ax1.twinx()
      ax1.set_title(weight, fontsize=16)
      # True Positives
      tp = TP[(TP[MODEL_STR] == model) & (TP[WEIGHT_STR] == weight)]
      # P True Positives
      ptp = tp[tp[PHASE_STR] == PWAVE]
      ptp = ptp[THRESHOLD_STR].value_counts().sort_index()
      # S True Positives
      stp = tp[tp[PHASE_STR] == SWAVE]
      stp = stp[THRESHOLD_STR].value_counts().sort_index()
      # False Negatives
      fn = FN[(FN[MODEL_STR] == model) & (FN[WEIGHT_STR] == weight)]
      # P False Negatives
      pfn = fn[fn[PHASE_STR] == PWAVE]
      pfn = pfn[THRESHOLD_STR].value_counts().sort_index()
      # S False Negatives
      sfn = fn[fn[PHASE_STR] == SWAVE]
      sfn = sfn[THRESHOLD_STR].value_counts().sort_index()
      pRECALL = ptp / (ptp + pfn)
      sRECALL = stp / (stp + sfn)
      RECALL = (ptp + stp) / ((ptp + pfn) + (stp + sfn))
      pRECALL.plot(ax=ax2, label=PWAVE, use_index=False, color="r")
      sRECALL.plot(ax=ax2, label=SWAVE, use_index=False, color="b")
      RECALL.plot(ax=ax2, label=f"{PWAVE} + {SWAVE}", use_index=False,
                  color="k")
      TPFN = pd.DataFrame({
        SPACE_STR.join([PWAVE, TP_STR]): ptp,
        SPACE_STR.join([SWAVE, TP_STR]): stp,
        SPACE_STR.join([PWAVE, FN_STR]): pfn,
        SPACE_STR.join([SWAVE, FN_STR]): sfn
      })
      TPFN.plot(kind='bar', ax=ax1, width=0.7)
      ax1.set(ylabel="Number of Picks", ylim=(0, m))
      ax2.set(ylim=(0, 1))
      yticks, yticklabels = ax2.get_yticks(), ax2.get_yticklabels()
      ax2.set(yticks=[], yticklabels=[])
      ax1.grid()
      ax1.get_legend().remove()
    axs[0].set(xlabel=None, xticklabels=[])
    axs[0].legend()
    axs[1].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])
    ax1 = axs[1].twinx()
    ax1.set(ylabel=RECALL_STR)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    axs[2].set()
    axs[3].set(xlabel=THRESHOLD_STR, ylabel=None, yticklabels=[])
    ax2.set_ylabel(RECALL_STR)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    ax2.legend()
    IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                    UNDERSCORE_STR.join(["TPFN", model]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()

  m = max(m, FP.groupby(groups)[THRESHOLD_STR].value_counts().max())
  m = (m + 9) // 10 * 10
  for model, phase in itertools.product(args.models, [PWAVE, SWAVE]):
    _, _axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = _axs.flatten()
    for ax1, weight in zip(axs, args.weights):
      ax2 = ax1.twinx()
      ax1.set_title(weight, fontsize=16)
      tp = TP[(TP[PHASE_STR] == phase) & (TP[WEIGHT_STR] == weight)]
      tp = tp[THRESHOLD_STR].value_counts().sort_index()
      fn = FN[(FN[PHASE_STR] == phase) & (FN[WEIGHT_STR] == weight)]
      fn = fn[THRESHOLD_STR].value_counts().sort_index()
      fp = FP[(FP[PHASE_STR] == phase) & (FP[WEIGHT_STR] == weight)]
      fp = fp[THRESHOLD_STR].value_counts().sort_index()
      PRECISION = tp / (tp + fp)
      RECALL = tp / (tp + fn)
      F1 = 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
      RECALL.plot(ax=ax2, label=RECALL_STR, use_index=False, color="b")
      F1.plot(ax=ax2, label=F1_STR, use_index=False, color="r")
      TPFNFP = pd.DataFrame({
        TP_STR: tp,
        FN_STR: fn,
        FP_STR: fp
      })
      TPFNFP.plot(kind='bar', ax=ax1, width=0.7)
      ax1.set(ylabel="Number of Picks", ylim=(1, m), yscale="log")
      ax2.set(ylim=(0, 1))
      yticks, yticklabels = ax2.get_yticks(), ax2.get_yticklabels()
      ax2.set(yticks=[], yticklabels=[])
      ax1.grid()
      ax1.get_legend().remove()
    axs[0].set(xlabel=None, xticklabels=[])
    axs[0].legend()
    axs[1].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])
    ax1 = axs[1].twinx()
    ax1.set_ylabel("Score")
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    axs[2].set()
    axs[3].set(xlabel=THRESHOLD_STR, ylabel=None, yticklabels=[])
    ax2.set_ylabel("Score")
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    ax2.legend()
    IMG_FILE = \
      Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
           UNDERSCORE_STR.join(["TPFNFP", model, phase]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
  return TP

def event_parser(filename : Path, args : argparse.Namespace) -> pd.DataFrame:
  """
  input  :
    - filename      (Path)
    - args          (argparse.Namespace)

  output :
    - pd.DataFrame

  errors :
    - FileNotFoundError

  notes  :
    | EVENT | STATION | PHASE | BEGDT | WEIGHT |
    --------------------------------------------
  """
  global DATA_PATH
  DATA_PATH = args.directory.parent
  WAVEFORMS_DATA = ini.waveform_table(args)
  # TODO: Stations are not considered due to the low amount of data
  SOURCE, DETECT = prs.event_parser(filename, *args.dates, None)
  TRUE_S = pd.DataFrame(columns=HEADER_SRC)
  TRUE_D = pd.DataFrame(columns=HEADER_MANL)
  for date, dataframe_d in WAVEFORMS_DATA.groupby(BEG_DATE_STR):
    start = UTCDateTime.strptime(date, DATE_FMT)
    end = start + ONE_DAY
    station = dataframe_d[STATION_STR].unique().tolist()
    if SOURCE is not None:
      source = SOURCE[SOURCE[TIMESTAMP_STR].between(start, end,
                                                    inclusive='left')]
      if not source.empty:
        TRUE_S = pd.concat([TRUE_S, source]) if not TRUE_S.empty else source
    TRUE_D = pd.concat([TRUE_D, DETECT[
      (DETECT[TIMESTAMP_STR].between(start, end, inclusive='left')) &
      (DETECT[STATION_STR].isin(station))]])
  if args.verbose:
    TRUE_S.to_csv(Path(DATA_PATH,
                       UNDERSCORE_STR.join([TRUE_STR, SOURCE_STR]) + CSV_EXT),
                       index=False)
    TRUE_D.to_csv(Path(DATA_PATH,
                       UNDERSCORE_STR.join([TRUE_STR, DETECT_STR]) + CSV_EXT),
                       index=False)
  return TRUE_S, TRUE_D

def time_displacement(DATA : pd.DataFrame, args : argparse.Namespace,
                      phase : str = PWAVE, method : str = "Picker") -> None:
  """
  input  :
    - DATA          (pd.DataFrame)
    - args          (argparse.Namespace)
    - phase         (str)

  output :
    - None

  errors :
    - AttributeError

  notes  :
    | MODEL | WEIGHT | PHASE | THRESHOLD | TIMESTAMP |
    --------------------------------------------------
  """
  global THRESHOLDS
  if args.verbose: print("Plotting the Time Displacement")
  DATA[TIMESTAMP_STR] = \
    DATA[TIMESTAMP_STR].map(lambda x: UTCDateTime(x[0]) - UTCDateTime(x[1]))
  DATA[PROBABILITY_STR] = DATA[PROBABILITY_STR].map(lambda x: x[1])
  bins = np.linspace(-0.5, 0.5, 21, endpoint=True)
  groups = [MODEL_STR, WEIGHT_STR, PHASE_STR]
  m = 0
  for (model, weight, phase), dtfrm in DATA.groupby(groups):
    counts, _ = np.histogram(dtfrm[TIMESTAMP_STR], bins=bins)
    m = max(m, max(counts))
  m = (m + 9) // 10 * 10
  for model, phase in itertools.product(args.models, [PWAVE, SWAVE]):
    fig, _axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = _axs.flatten()
    dataframe_mp = DATA[(DATA[MODEL_STR] == model) &
                        (DATA[PHASE_STR] == phase)].reset_index(drop=True)
    for ax, (weight, dtfrm) in zip(axs, dataframe_mp.groupby(WEIGHT_STR)):
      ax.set_title(weight)
      counts, _ = np.histogram(dtfrm[TIMESTAMP_STR], bins=bins)
      mu = np.mean(dtfrm[TIMESTAMP_STR])
      std = np.std(dtfrm[TIMESTAMP_STR])
      ax.bar(bins[:-1], counts, label=rf"$\mu$={mu:.2f},$\sigma$={std:.2f}",
             alpha=0.5, width=0.05)
      for t_i, t_f in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
        data = dtfrm[dtfrm[PROBABILITY_STR].between(t_i, t_f, inclusive='left')
                    ][TIMESTAMP_STR]
        # TODO: Consider a KDE plot
        counts, _ = np.histogram(data, bins=bins)
        ax.step(bins[:-1], counts, where='mid', label=rf"[{t_i},{t_f})")
      data = dtfrm[
               dtfrm[PROBABILITY_STR] >= THRESHOLDS[-1]][TIMESTAMP_STR]
      counts, _ = np.histogram(data, bins=bins)
      ax.step(bins[:-1], counts, where='mid', label=rf"[{THRESHOLDS[-1]},1)")
      ax.set(xlim=(-0.5, 0.5), ylim=(0, m))
      ax.grid()
      ax.legend()
    xlabel = "Time Displacement (s)"
    ylabel = f"Number of {phase} picks"
    axs[0].set(xlabel=None, xticklabels=[], ylabel=ylabel)
    axs[1].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])
    axs[2].set(xlabel=xlabel, ylabel=ylabel)
    axs[3].set(xlabel=xlabel, ylabel=None, yticklabels=[])
    IMG_FILE = Path(IMG_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                    UNDERSCORE_STR.join([method, TIME_DSPLCMT_STR, model,
                                         phase]) + PNG_EXT)
    plt.tight_layout()
    plt.savefig(IMG_FILE)
    plt.close()

def main(args : argparse.Namespace):
  global DATA_PATH, DATES
  DATA_PATH : Path = args.directory.parent
  # Picker
  ANALYSIS = "Picker"
  PRED = ini.classified_loader(args)
  if args.verbose:
    PRED.to_csv(Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                     UNDERSCORE_STR.join([ANALYSIS, PRED_STR]) + CSV_EXT),
                     index=False)
  if args.file is None: raise ValueError("No event file given")
  TRUE_S, TRUE_D = event_parser(args.file, args)
  TRUE_D = TRUE_D[TRUE_D[STATION_STR].isin(PRED[STATION_STR].unique())]
  # plot_data(copy.deepcopy(TRUE_D), copy.deepcopy(PRED), args)
  # plot_data(copy.deepcopy(TRUE_D), copy.deepcopy(PRED), args, phase=SWAVE)
  TP = stat_test(copy.deepcopy(TRUE_D), copy.deepcopy(PRED), args, ANALYSIS)
  if args.verbose:
    TP.to_csv(Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                   UNDERSCORE_STR.join([ANALYSIS, TP_STR]) + CSV_EXT),
                   index=False)
  time_displacement(copy.deepcopy(TP), args)
  # Associator
  ANALYSIS = "GaMMA"
  PRED = ini.associated_loader(args)
  if args.verbose:
    PRED.to_csv(Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                     UNDERSCORE_STR.join([ANALYSIS, PRED_STR]) + CSV_EXT),
                     index=False)
  start, end = args.dates
  TP = pd.DataFrame([], columns=HEADER_PRED)
  if DATES is None:
    DATES = [start]
    while DATES[-1] <= end: DATES.append(DATES[-1] + ONE_DAY)
  for s, e in zip(DATES[:-1], DATES[1:]):
    PRE = PRED[PRED[TIMESTAMP_STR].between(s, e, inclusive='left')]
    if PRE.empty: continue
    TP = pd.concat([TP, stat_test(copy.deepcopy(TRUE_D), copy.deepcopy(PRE),
                                  args, ANALYSIS)])
  if args.verbose:
    TP.to_csv(Path(DATA_PATH, ("D_" if args.denoiser else EMPTY_STR) + \
                   UNDERSCORE_STR.join([ANALYSIS, TP_STR]) + CSV_EXT),
                   index=False)
  time_displacement(copy.deepcopy(TP), args)

if __name__ == "__main__": main(ini.parse_arguments())