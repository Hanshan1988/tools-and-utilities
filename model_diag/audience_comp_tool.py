import pandas as pd
import numpy as np
import os
import sys
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess as sp
from subprocess import Popen, PIPE
import logging
import decimal
import math
from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# LOC_SCR_DIR = 'aud_comp'
# # DATA_FNAME = 'aud_comp_data.parquet'
# MDL_FNAME = 'aud_comp_model.pickle'
# DATE_FMT    = '%Y-%m-%d'
# date_prefix = datetime.now().strftime(DATE_FMT)

# # remove_dir(LOC_SCR_DIR)
# create_dir(LOC_SCR_DIR)

class AudienceCompTool():
    def __init__(self, update_local=False, proj_id='wx-bq-poc', get_label=True,
                 label_table='loyalty_bi_analytics.fact_campaign_sales', 
                 loc_scr_dir='aud_comp', mdl_fname='model_object.pickle',
                 l_data_path=[], camp_pref='CVM', l_groupby_col=[],
                 l_score_col=[], gs_model_loc='', inc_feat_cols=False, n_top=5):
        """
        Gets the scored data from gcs locations
        This does not need ground truth labels
        :local:       if local then data files are assumes to be stored locally - else transfer from gs locations
        :l_data_path: a list of locations pointing to the data
        :camp_pref:   the prefix of campaigns included in the locations of l_data_path
        :get_label:    whether to query bq to find label data
        """
        self.l_data_path = l_data_path
        self.num_dataset = len(self.l_data_path)
        self.get_label = get_label
        self.l_groupby_col = l_groupby_col
        self.l_score_col = l_score_col
        self.label_table = label_table if get_label else ''
        self.proj_id = proj_id
        self.loc_scr_dir = loc_scr_dir
        
        if len(camp_pref) > 0: 
            self.l_camp_name = [re.search(f'({camp_pref}-\d+)', data_path).group(0) for data_path in l_data_path]
        else:
            self.l_camp_name = list(range(self.num_dataset))
        self.l_datafname = [f'aud_comp_data_{camp_name}.parquet' for camp_name in self.l_camp_name]
        
        str_l_camp_name = ', '.join(self.l_camp_name )
        print(f'There are {self.num_dataset} datasets corresponding to campaign names: {str_l_camp_name}')
        
        for gs_data_loc, data_fname in zip(self.l_data_path, self.l_datafname):
            gs_data_pq_loc = f'{gs_data_loc}/*.pq'
            fname = f'{loc_scr_dir}/{data_fname}'
            if not os.path.exists(fname) or update_local:
#                 copy_file(gs_data_loc, fname, '*.pq') 
                print(f'Copying data from {gs_data_pq_loc} to {fname} ...')
                sp.call(f'gsutil -m cp {gs_data_pq_loc} {fname}', shell=True) 
                # assumes we only have one pq file in a particular gs_data_loc
                # how about partitioned files? 
                # Get the file with most recent date format YYYY-MM-DD in the file name

        # Only if we want to look at features as well 
        if len(gs_model_loc) > 0: # if we have a location of model pickle
            copy_file(gs_model_loc, f'{loc_scr_dir}/{mdl_fname}', 'model_object.pickle')

            # read in model and get top features
            model = unpickle(f'{loc_scr_dir}/{MDL_FNAME}')
            df_var_imp_sorted = model.varimp.sort_values('avg_rel_imp', ascending=False)
            df_var_imp_top_n = df_var_imp_sorted.head(n_top)
            top_n_feats = df_var_imp_top_n.index.tolist()

        # read in data and get relevant columns
        l_id_col = ['crn', 'dataset_name']
        l_feat_col = top_n_feats if inc_feat_cols else []
        l_aux_cols = ['model', 'selection_group']
        incl_cols = list(set(l_id_col + l_score_col + l_feat_col + l_groupby_col + l_aux_cols))

        df = pd.DataFrame()
        for data_fname, name in zip(self.l_datafname, self.l_camp_name):
            df_dataset = pd.read_parquet(f'{loc_scr_dir}/{data_fname}')
            df_dataset['dataset_name'] = name
            df = pd.concat([df, df_dataset], axis=0).reset_index(drop=True)
        # check assumptions
        assert all([col in df for col in incl_cols]), "Not all columns are in dataframe"
        assert all(df.groupby(['dataset_name', 'crn'])['crn'].count() == 1), "One campaign may have non-distinct crns"
        self.df = df[incl_cols].reset_index(drop=True)
        print('List of columns:', ', '.join(self.df.columns.tolist()))
        
    def get_label_data(self, auto_dedupe_cap=.001, update_local=False):
        
        str_l_camp_name = f"""({', '.join([f"'{camp}'" for camp in self.l_camp_name])})"""
        local_fname = f'{self.loc_scr_dir}/aud_comp_label.parquet'
        # assume distinct crn by campaign_code and campaign_start_date
        query = f"""
            with no_grp as (
            select fcs.crn, fcs.campaign_code, fcs.campaign_name, fcs.campaign_start_date
              ,MIN(fcs.offer_start_date) AS offer_start_date
              ,MAX(fcs.offer_end_date) AS offer_end_date
              ,MAX(CASE WHEN fcs.send_flag IS NOT NULL THEN 1 ELSE 0 END) AS send_flag
              ,MAX(CASE WHEN fcs.open_flag IS NOT NULL THEN 1 ELSE 0 END) AS open_flag
              ,MAX(CASE WHEN fcs.activate_flag IS NOT NULL THEN 1 ELSE 0 END) AS activate_flag
              ,MAX(CASE WHEN fcs.redeem_flag IS NOT NULL THEN 1 ELSE 0 END) AS redeem_flag
              ,MAX(CASE WHEN fcs.shop_online_flag IS NOT NULL THEN 1 ELSE 0 END) AS shop_online_flag
              ,MAX(CASE WHEN oh.customer_pickup_flag = 'T' THEN 1 ELSE 0 END) AS shop_pickup_flag
              ,MAX(CASE WHEN oh.customer_pickup_flag = 'F' THEN 1 ELSE 0 END) AS shop_delivery_flag
            FROM {self.label_table} fcs
            -- loyalty card detail
            LEFT JOIN loyalty.lylty_card_detail as lcd
              ON fcs.crn = lcd.crn
            -- pickup flag (@todo: replace with loyalty_bi_analytics.online_order_pickup)
            LEFT JOIN loyalty.basket_sales_summary_cust_order AS op
              ON lcd.lylty_card_nbr = op.lylty_card_nbr
              AND op.start_txn_date BETWEEN fcs.offer_start_date AND fcs.offer_end_date
            -- order number mapping
            LEFT JOIN loyalty.mor_smkt_ecf_order_header AS oh
              ON op.order_nbr = CAST(oh.order_no AS STRING)
            where fcs.campaign_code in {str_l_camp_name}
              -- and send_flag is not null
            group by 
              1,2,3,4
            )

          -- get groups, groups are not necessary for labels as we can use scored groups
          SELECT no_grp.*
            ,grp.group01, grp.group02, grp.group03, grp.group04, grp.group05, grp.group06
          FROM no_grp 
          LEFT JOIN loyalty_bi_analytics.fact_campaign_split grp
            ON no_grp.crn = grp.crn 
            AND no_grp.campaign_start_date = grp.campaign_start_date 
            AND no_grp.campaign_code = grp.campaign_code
            AND grp.campaign_code in {str_l_camp_name}
        """
        print(query)
        if not update_local:
            if os.path.exists(local_fname):
                self.df_label = pd.read_parquet(local_fname)
            else:
                self.df_label = pd.read_gbq(query, project_id=self.proj_id, dialect='standard')
                self.df_label.to_parquet(local_fname)
        else:
            self.df_label = pd.read_gbq(query, project_id=self.proj_id, dialect='standard')
            self.df_label.to_parquet(local_fname)
            
        # duplicate CRN-campaign-start_date combo
        groupby_cols = ['campaign_code', 'campaign_start_date', 'crn']
        crn_count = self.df_label.groupby(groupby_cols)['crn'].count()
        if not all(crn_count == 1):
            logging.warning("Campaign-date combos may have non-distinct crns")
            df_reindex = self.df_label.set_index(groupby_cols)
            multi_idx_gt_1 = crn_count[crn_count > 1]
            df_label_gt_1 = df_reindex[df_reindex.index.isin(multi_idx_gt_1.index)].reset_index()
            df_label_gt_1.to_csv(f'{self.loc_scr_dir}/aud_comp_label_crn_gt_1.csv', index=False)
            if df_label_gt_1.shape[0] / self.df_label.shape[0] < auto_dedupe_cap:
                str_join = '+'.join(groupby_cols)
                print(f'Number of rows before dedupe on {str_join}:{self.df_label.shape[0]}')
                self.df_label.drop_duplicates(subset=groupby_cols)
                print(f'Number of rows after dedupe on {str_join} :{self.df_label.shape[0]}')
    
    def merge_w_label_data(self, update_local=False):
        if self.get_label:
            print("Getting Label Data from BQ!")
            self.get_label_data(update_local=update_local)
            # checks
            for name in self.l_camp_name:
                df_camp_scored = self.df[self.df['dataset_name'] == name]
                rows_scored = df_camp_scored.shape[0]
                crn_scored = set(df_camp_scored.crn)
                df_camp_label = self.df_label[self.df_label['campaign_code'] == name]
                rows_label = df_camp_label.shape[0]
                crn_label = set(df_camp_label.crn)
                crn_intersect = crn_scored & crn_label
                crn_scored_minus_label = crn_scored - crn_intersect
                crn_label_minus_scored = crn_label - crn_intersect
                print(name)
                print(f'Scored data:  {rows_scored} rows {len(crn_scored)} crns, {100*len(crn_intersect)/len(crn_scored) :.2f}% overlap')
                print(f'Label data :  {rows_label} rows {len(crn_label)} crns, {100*len(crn_intersect)/len(crn_label) :.2f}% overlap')
                print(f'Overlap data: {len(crn_intersect)} crns')
                
                self.df_merge = pd.merge(self.df, self.df_label, how='left', 
                                 left_on=['crn', 'dataset_name'], right_on = ['crn', 'campaign_code'])
                self.df_overlap = self.df_merge[pd.notnull(self.df_merge['campaign_code'])].reset_index(drop=True)
        else:
            print("Dry run: NO Label Data!")
            self.df_merge = self.df
            self.df_overlap = self.df
            self.df_scored_sent = self.df[self.df.selection_group != 'do not send'].reset_index(drop=True)
                                 
    def get_audience_comp_summary(self):
        l_crn = [set(self.df[self.df['dataset_name'] == name].crn) for name in self.l_camp_name]

        # base case assumes length 1 at least
        s_intersect = l_crn[0]
        for i in range(1, len(l_crn)):
            s_intersect = s_intersect & l_crn[i]
        l_intersect = list(s_intersect)
        for name, i in zip(self.l_camp_name, range(len(l_crn))):   
            print(f'Dataset {name}: {round(100 * len(s_intersect) / len(l_crn[i]), 1)}% intersect')
        
        df_summ = self.df.groupby(self.l_groupby_col)[self.l_score_col].agg(['count', 'min', 'mean', 'median', 'max'])
        return df_summ
    
    def get_audience_comp_detail(self):
        # assume ordered list: check between pair of campaigns
        # TODO: Some contains NEM whereas others don't
        l_df = [self.df_scored_sent[self.df_scored_sent['dataset_name'] == name].reset_index(drop=True) for name in self.l_camp_name]
        for i in range(len(self.l_camp_name) - 1):
            df_c1 = l_df[i]
            df_c2 = l_df[i + 1]
            compare_two_cohorts(df_c1, df_c2, send_col='', redeem_col='')
    
    def get_model_summary(df, y_true_col, y_score_col, l_groupby_cols=[], clf=True):
        if clf:
            measure_1 = roc_auc_score(df[y_true_col], df[y_score_col])
        else: # regression
            measure_1 = mean_squared_error(df[y_true_col], df[y_score_col])
        return measure_1

    def get_psi_values():
        """
        Plots of distribution changes can be done using plot_aud_hist_comp
        This function gets psi values
        """
        pass
    
    def plot_model_perf_change(l_ts, d_metrics):
        """
        Plots how model performance changes over time
        Can be either classification or regression metrics
        l_ts is a list of datetime
        d_metrics is a dictionary metric names as keys and lists of values for the metric as values
        """
        # convert into dataframe
        # TODO: different y-axis
        df_plot = pd.DataFrame(d_metrics)
        df_plot = df_plot.set_index(l_ts)
        df_plot.sort_index(inplace=True)

        plot_2d_ts(df_plot, xlab='Date', filename = '2d-ts.html')


def find_group_cols(df):
    """
    Find groups from loyalty_bi_analytics.fact_campaign_split
    """
    cols = [f'group0{i}' for i in range(1, 7)]
    d_uniq_vals = {col:list(set(df[col])) for col in cols}
    d_excl_none = {k:[i for i in v if i] for k, v in d_uniq_vals.items()}
    d_str_join = {k:'|'.join(v).lower() for k, v in d_excl_none.items()}
    # marketable vs non-marketable
    # acquisition vs others
    # random vs pickup vs delivery vs nem
    d_out = {}
    for k, v in d_str_join.items():
        if 'pick' in v and 'deliver' in v and 'random' in v:
            d_out['model_type'] = k
        elif 'acquisition' in v or 'rns' in v:
            d_out['acq_type'] = k
        elif 'marketable' in v and 'non' in v:
            d_out['marketable_type'] = k
    return d_out

def apply_parallel(df, groupby_col, func):
    df_grouped = df.groupby(groupby_col)
    groups = [group_val for group_val, group in df_grouped]
    l = Parallel(n_jobs=mp.cpu_count())(delayed(func)(group) for _, group in df_grouped)
    x = l[0]
    if isinstance(x, pd.DataFrame):
        return pd.concat(l_df)
    else:
        return list(zip(groups, l))

def roc_auc_score_func(df, y_true_col='y_true', y_score_col='y_score'):
    measure = roc_auc_score(df[y_true_col].values, df[y_score_col].values)
    measure = round(measure, 4)
    return measure

def mean_squared_error_func(df, y_true_col='y_true', y_score_col='y_score'):
    measure = mean_squared_error(df[y_true_col].values, df[y_score_col].values)
    measure = round(measure, 4)
    return measure

def accuracy_score_func(df, y_true_col='y_true', y_score_col='y_score'):
    measure = accuracy_score(df[y_true_col].values, df[y_score_col].values)
    measure = round(measure, 4)
    return measure

def response_rate_func(df, y_true_col='y_true'):
    # response rate in percentage
    measure = df[y_true_col].sum() / df[y_true_col].count()
    measure = round(100 * measure, 2)
    return measure

def get_model_summary(df, l_groupby_cols, func, clf=True, **kwargs):
    df_measure_1 = apply_parallel(df, l_groupby_cols, lambda df: func(df, **kwargs))
    return df_measure_1

def run_py_script_on_data(set_py_path, py_script_path, data_path, verbose=False):
    str_sp_score = f'{set_py_path} python {py_script_path} {data_path}'
    # python script path need to end with dist_udf.py
    
    print(str_sp_score)
    
    if not verbose:
        sp.call(str_sp_score, shell=True)
    else:
        p = Popen(str_sp_score, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
        print(str(output, 'utf-8'))

def compare_two_cohorts(df_1, df_2, label=False, camp_col ='dataset_name', excl_grp_val='do not send',
                        slct_grp_col='selection_group', mdl_col='model', crn_col='crn', send_col='', redeem_col=''):
    """
    The two dataframes would be broken down by 'selection_group' column
    Assume redemption rate is small in both cases and all data not in 'do not send' group is sent
    Assume the timeline is from df_1 to df_2 chronologically
    """
    
    l_common_cols = set(df_1.columns) & set(df_2.columns)
    l_check_cols = set([camp_col, crn_col, send_col, redeem_col, slct_grp_col]) if label \
        else set([camp_col, crn_col, slct_grp_col])
    df_1_send = df_1[df_1[slct_grp_col] != excl_grp_val].reset_index(drop=True)
    df_2_send = df_2[df_2[slct_grp_col] != excl_grp_val].reset_index(drop=True)
#     send_val_counts_1 = df_1_send[slct_grp_col].value_counts(normalize=True)
#     send_val_counts_2 = df_2_send[slct_grp_col].value_counts(normalize=True)
    df_1_send_nonrand = df_1[df_1[slct_grp_col] != 'random'].reset_index(drop=True)
    df_2_send_nonrand = df_2[df_2[slct_grp_col] != 'random'].reset_index(drop=True)
    
    camp_1 = df_1_send[camp_col].unique()[0]
    camp_2 = df_2_send[camp_col].unique()[0]
    # First split between marketable and non-marketable from mdl_col
    df_1_s_nonrand_mkt  = df_1_send_nonrand[df_1_send_nonrand[mdl_col] != 'non-marketable'].reset_index(drop=True)
    df_1_s_nonrand_nmkt = df_1_send_nonrand[df_1_send_nonrand[mdl_col] == 'non-marketable'].reset_index(drop=True)
    df_2_s_nonrand_mkt  = df_2_send_nonrand[df_2_send_nonrand[mdl_col] != 'non-marketable'].reset_index(drop=True)
    df_2_s_nonrand_nmkt = df_2_send_nonrand[df_2_send_nonrand[mdl_col] == 'non-marketable'].reset_index(drop=True)
    # TODO: Then split between marketable pickup and marketable delivery if available from slct_grp_col

    if len(l_common_cols & l_check_cols) == len(l_check_cols):
        l_crn_s_1 = set(df_1_send[crn_col])
        l_crn_s_2 = set(df_2_send[crn_col])
        l_crn_s_1_or_2 = l_crn_s_1 | l_crn_s_2
        l_crn_s_1_and_2 = l_crn_s_1 & l_crn_s_2
        l_crn_s_1_not_2 = l_crn_s_1 - l_crn_s_2
        l_crn_s_2_not_1 = l_crn_s_2 - l_crn_s_1
        
        l_crn_nonrand_1 = set(df_1_send_nonrand[crn_col])
        l_crn_nonrand_2 = set(df_2_send_nonrand[crn_col])
        l_crn_nonrand_1_not_2 = l_crn_nonrand_1 - l_crn_nonrand_2
        l_crn_nonrand_2_not_1 = l_crn_nonrand_2 - l_crn_nonrand_1
        
        l_crn_s_1_mkt = set(df_1_s_nonrand_mkt[crn_col])
        l_crn_s_1_nmkt = set(df_1_s_nonrand_nmkt[crn_col])
        l_crn_s_2_mkt = set(df_2_s_nonrand_mkt[crn_col])
        l_crn_s_2_nmkt = set(df_2_s_nonrand_nmkt[crn_col])
        
        pct_s_1_rand = round(100 * (1 - len(l_crn_nonrand_1) / len(l_crn_s_1)), 1)
        pct_s_1_mkt  = round(100 * len(l_crn_s_1_mkt) / len(l_crn_s_1), 1)
        pct_s_1_nmkt = round(100 * len(l_crn_s_1_nmkt) / len(l_crn_s_1), 1)
        pct_s_2_rand = round(100 * (1 - len(l_crn_nonrand_2) / len(l_crn_s_2)), 1)
        pct_s_2_mkt  = round(100 * len(l_crn_s_2_mkt) / len(l_crn_s_2), 1)
        pct_s_2_nmkt = round(100 * len(l_crn_s_2_nmkt) / len(l_crn_s_2), 1)
        
        print(f'==========Comparing {camp_1} VS {camp_2}==========')
        print(f'{camp_1} has {len(l_crn_s_1)} crns:',
              f'rand {pct_s_1_rand} pct, mkt {pct_s_1_mkt} pct, nmkt {pct_s_1_nmkt} pct')
        print(f'{camp_2} has {len(l_crn_s_2)} crns:',
              f'rand {pct_s_2_rand} pct, mkt {pct_s_2_mkt} pct, nmkt {pct_s_2_nmkt} pct')
        
        l_crn_mkt_1_and_2 = l_crn_s_1_mkt & l_crn_s_2_mkt
        l_crn_nmkt_1_and_2 = l_crn_s_1_nmkt & l_crn_s_2_nmkt
        
        l_crn_mkt_1_not_2 = l_crn_s_1_mkt & l_crn_nonrand_1_not_2
        l_crn_nmkt_1_not_2 = l_crn_s_1_nmkt & l_crn_nonrand_1_not_2
        
        l_crn_mkt_1_nmkt_2 = l_crn_s_1_mkt & l_crn_s_2_nmkt
        l_crn_mkt_2_nmkt_1 = l_crn_s_2_mkt & l_crn_s_1_nmkt
        
        l_crn_mkt_2_not_1 = l_crn_s_2_mkt  & l_crn_nonrand_2_not_1
        l_crn_nmkt_2_not_1 = l_crn_s_2_nmkt & l_crn_nonrand_2_not_1
        
#         print(len(l_crn_s_1_mkt), len(l_crn_mkt_1_and_2) + len(l_crn_mkt_1_nmkt_2) + len(l_crn_mkt_1_not_2))
#         print(len(l_crn_s_1_nmkt), len(l_crn_mkt_2_nmkt_1) + len(l_crn_nmkt_1_and_2) + len(l_crn_nmkt_1_not_2))
#         print(len(l_crn_nonrand_2_not_1), len(l_crn_mkt_2_not_1) + len(l_crn_nmkt_2_not_1))
#         print(len(l_crn_s_2_mkt), len(l_crn_mkt_1_and_2) + len(l_crn_mkt_2_nmkt_1) + len(l_crn_mkt_2_not_1))
#         print(len(l_crn_s_2_nmkt), len(l_crn_mkt_1_nmkt_2) + len(l_crn_nmkt_1_and_2) + len(l_crn_nmkt_2_not_1))
#         print(len(l_crn_nonrand_1_not_2), len(l_crn_mkt_1_not_2) + len(l_crn_nmkt_1_not_2))
        
        if send_col != '' and redeem_col != '':
            df_1_redm = df_1_send[df_1_send[redeem_col] == 1].reset_index(drop=True)
            df_2_redm = df_2_send[df_2_send[redeem_col] == 1].reset_index(drop=True)
            l_crn_r_1 = set(df_1_redm[crn_col])
            l_crn_r_2 = set(df_2_redm[crn_col])
            l_crn_nr_1 = l_crn_s_1 - l_crn_r_1
            l_crn_nr_2 = l_crn_s_2 - l_crn_r_2
            pct_r_1 = round(100 * len(l_crn_r_1) / len(l_crn_s_1), 1)
            pct_r_2 = round(100 * len(l_crn_r_2) / len(l_crn_s_2), 1)
            # make up of redeem 2
            l_crn_r_2_from_nr_1 = l_crn_r_2 & l_crn_nr_1 # from not redeem 1
            l_crn_r_2_not_from_s_1 = l_crn_r_2 - l_crn_s_1_not_2
            # make up of not redeem 2
            l_crn_nr_2_from_nr_1 = l_crn_nr_2 & l_crn_nr_1 # from not redeem 1
            l_crn_nr_2_not_from_s_1 = l_crn_nr_2 - l_crn_s_1
            
        pct_s_1_or_2_by_1 = round(100 * len(l_crn_s_1_or_2) / len(l_crn_s_1), 1)
        pct_s_1_or_2_by_2 = round(100 * len(l_crn_s_1_or_2) / len(l_crn_s_2), 1)
        pct_s_1_and_2_by_1 = round(100 * len(l_crn_s_1_and_2) / len(l_crn_s_1), 1)
        pct_s_1_and_2_by_2 = round(100 * len(l_crn_s_1_and_2) / len(l_crn_s_2), 1)
        pct_s_1_not_2_by_1 = round(100 * len(l_crn_s_1_not_2) / len(l_crn_s_1), 1)
        pct_s_2_not_1_by_2 = round(100 * len(l_crn_s_2_not_1) / len(l_crn_s_2), 1)
        print(f'{camp_1} send has {len(l_crn_s_1_and_2)} crns in {camp_2} ({pct_s_1_and_2_by_1} pct of old campaign goes on to new campaign)')
        print(f'{camp_2} send has {len(l_crn_s_2_not_1)} crns NOT in {camp_1} ({pct_s_2_not_1_by_2} pct of new campaign are new customers)')
        
        # 0 - C1 marketable, 1 - C1 non-marketable, 2 - in C2 not C1
        # 3 - C2 marketable, 4 - C2 non-marketable, 5 - in C1 not C2
    
        ploty_92_colours = px.colors.named_colorscales()
        ploty_10_colours = px.colors.qualitative.Plotly
        color_10 = ["rgba(31, 119, 180, 0.8)", "rgba(255, 127, 14, 0.8)",
                    "rgba(44, 160, 44, 0.8)", "rgba(214, 39, 40, 0.8)",
                    "rgba(148, 103, 189, 0.8)", "rgba(140, 86, 75, 0.8)",
                    "rgba(227, 119, 194, 0.8)", "rgba(127, 127, 127, 0.8)",
                    "rgba(188, 189, 34, 0.8)", "rgba(23, 190, 207, 0.8)"]
        
        node_label = ["C1 Marketable", "C1 Non-marketable", "In C2 not in C1", 
                      "C2 Marketable", "C2 Non-marketable", "In C1 not in C2"]

        source = [0, 0, 0] + [1, 1, 1] + [2, 2]
        target = [3, 4, 5] + [3, 4, 5] + [3, 4]
        value = [len(l_crn_mkt_1_and_2), len(l_crn_mkt_1_nmkt_2), len(l_crn_mkt_1_not_2), 
                 len(l_crn_mkt_2_nmkt_1), len(l_crn_nmkt_1_and_2), len(l_crn_nmkt_1_not_2),
                 len(l_crn_mkt_2_not_1), len(l_crn_nmkt_2_not_1)]
       
        pct_of_l = [v/len(l_crn_nonrand_1) for v in value]
        pct_of_r = [v/len(l_crn_nonrand_2) for v in value]
        
        # Mark in C2 but not C1 as translucent
        node_color = color_10[:len(source)]
        node_color = 2 * ["rgba(31, 119, 180, 0.8)", "rgba(0, 0, 96, 0.4)", "rgba(200, 200, 200, 0.1)"] 
        link_opacity = '0.4'
        link_color = ["rgba(31, 119, 180, 0.4)"] * 3 + ["rgba(0, 0, 96, 0.4)"] * 3 + ["rgba(200, 200, 200, 0.1)"] * 2
        link_label = [f'Represents {round(l * 100, 1)}% of C1 and {round(r * 100, 1)}% of C2' for l, r in zip(pct_of_l, pct_of_r)]
        
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = node_label,
              color = node_color
            ),
            link = dict(
              source = source,
              target = target,
              value = value,
              color = link_color,
              label = link_label
        
          ))])

        fig.update_layout(title_text=f"{camp_1} VS {camp_2} Audience Comparison Sankey Diagram", font_size=12)
        plot(fig, filename=f'{camp_1}_{camp_2}_sankey.html')
        
