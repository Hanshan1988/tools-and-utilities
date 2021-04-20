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
                 l_data_path=[], camp_pref='CVM', l_groupby_col=[],
                 l_score_col=[], gs_model_loc='', inc_feat_cols=False, n_top=5):
        """
        This does not need ground truth labels
        :local:       if local then data files are assumes to be stored locally - else transfer from gs locations
        :l_data_path: a list of locations pointing to the data
        :camp_pref:   the prefix of campaigns included in the locations of l_data_path
        :get_label:    whether to query bq to find label data
        """
        self.l_data_path = l_data_path
        self.num_dataset = len(self.l_data_path)
        self.l_groupby_col = l_groupby_col
        self.l_score_col = l_score_col
        self.label_table = label_table if get_label else ''
        self.proj_id = proj_id
        
        if len(camp_pref) > 0: 
            self.l_camp_name = [re.search(f'({camp_pref}-\d+)', data_path).group(0) for data_path in l_data_path]
        else:
            self.l_camp_name = list(range(self.num_dataset))
        self.l_datafname = [f'aud_comp_data_{camp_name}.parquet' for camp_name in self.l_camp_name]
        
        str_l_camp_name = ', '.join(self.l_camp_name )
        print(f'There are {self.num_dataset} datasets corresponding to campaign names {str_l_camp_name}')
        
        for gs_data_loc, data_fname in zip(self.l_data_path, self.l_datafname):
            fname = f'{LOC_SCR_DIR}/{data_fname}'
            if not os.path.exists(fname) or update_local:
#                 copy_file(gs_data_loc, fname, '*.pq') 
                sp.process(f'cp {gs_data_loc} {fname}', shell=True) 
                # assumes we only have one pq file in a particular gs_data_loc
                # how about partitioned files? 
                # Get the file with most recent date format YYYY-MM-DD in the file name

        # Only if we want to look at features as well 
        if len(gs_model_loc) > 0: # if we have a location of model pickle
            copy_file(gs_model_loc, f'{LOC_SCR_DIR}/{MDL_FNAME}', 'model_object.pickle')

            # read in model and get top features
            model = unpickle(f'{LOC_SCR_DIR}/{MDL_FNAME}')
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
            df_dataset = pd.read_parquet(f'{LOC_SCR_DIR}/{data_fname}')
            df_dataset['dataset_name'] = name
            df = pd.concat([df, df_dataset], axis=0).reset_index(drop=True)
        # check assumptions
        assert all([col in df for col in incl_cols]), "Not all columns are in dataframe"
        assert all(df.groupby(['dataset_name', 'crn'])['crn'].count() == 1), "One campaign may have non-distinct crns"
        self.df = df[incl_cols].reset_index(drop=True)
        print(self.df.columns)
        
    def get_label_data(self, auto_dedupe_cap=.001,update_local=False):
        
        str_l_camp_name = f"""({', '.join([f"'{camp}'" for camp in self.l_camp_name])})"""
        local_fname = f'{LOC_SCR_DIR}/aud_comp_label.parquet'
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

          -- get groups
          select no_grp.*,
            grp.group01, grp.group02, grp.group03, grp.group04, grp.group05, grp.group06
          from no_grp 
          inner join loyalty_bi_analytics.fact_campaign_split grp
            on no_grp.crn = grp.crn 
            and no_grp.campaign_start_date = grp.campaign_start_date 
            and no_grp.campaign_code = grp.campaign_code
            and grp.campaign_code in {str_l_camp_name}
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
            df_label_gt_1.to_csv(f'{LOC_SCR_DIR}/aud_comp_label_crn_gt_1.csv', index=False)
            if df_label_gt_1.shape[0] / self.df_label.shape[0] < auto_dedupe_cap:
                str_join = '+'.join(groupby_cols)
                print(f'Number of rows before dedupe on {str_join}:{self.df_label.shape[0]}')
                self.df_label.drop_duplicates(subset=groupby_cols)
                print(f'Number of rows after dedupe on {str_join}: {self.df_label.shape[0]}')
    
    def get_model_perf(self, update_local=False):
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
            print(f'Overlap data: {len(crn_intersect)} crns' )
        self.df_merge = pd.merge(self.df, self.df_label, how='left', 
                                 left_on=['crn', 'dataset_name'], right_on = ['crn', 'campaign_code'])
        self.df_overlap = self.df_merge[pd.notnull(self.df_merge['campaign_code'])]
                                 
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
    
    def get_model_summary(df, y_true_col, y_score_col, l_groupby_cols=[], clf=True):
        if clf:
            measure_1 = roc_auc_score(df[y_true_col], df[y_score_col])
        else: # regression
            measure_1 = mean_squared_error(df[y_true_col], df[y_score_col])
        return measure_1
    
    def find_min_max_rng(df, val_col, normalise=False):
        # find the min and max values
        min_val = df[val_col].min() if not normalise else 0
        max_val = df[val_col].max() if not normalise else 1
        rng = max_val - min_val
        return min_val, max_val, rng

    def get_interval(df, val_col, min_val, max_val, bins):
        cut = pd.cut(np.array([min_val, max_val]), bins, retbins=True)
        cut_intervals = cut[1]
        bins_cut = pd.IntervalIndex.from_breaks(cut_intervals)
        s_interval = pd.cut(df[val_col], bins_cut)
        return s_interval

    def prep_data_for_hist(df, l_groupby_col, plot_col, x_normalize=True, bins=100):
        """
        Find the distribution of values for plot_col when grouped by l_groupby_col
        It's often hard to plot when we have more than 10k points
        if normalize: theoretical min is zero and max is one
        """
        # find the min and max values
        min_val, max_val, rng = find_min_max_rng(df, plot_col, True)
        precision = rng / bins
        s_interval = get_interval(df, plot_col, min_val, max_val, bins=100) # initial bins for cut-off
        
        cut = pd.cut(np.array([min_val, max_val]), 100, retbins=True)
        cut_intervals = cut[1]
        bins = pd.IntervalIndex.from_breaks(cut_intervals)
        
        df['interval_right'] = [i.right for i in s_interval.values]
        l_new_groupby = l_groupby_col + ['interval_right']
        df_plot = df.groupby(l_new_groupby)[plot_col].count().reset_index()

        # rename columns
        df_plot.columns = l_groupby_col + ['value', 'count']

        df_plot_norm = (df_plot.groupby(l_groupby_col + ['value'])['count'].apply(lambda x: x.sum()) / \
            df_plot.groupby(l_groupby_col)['count'].apply(lambda x: x.sum())).reset_index()
        df_plot_norm['cum_count_norm'] = df_plot_norm.groupby(l_groupby_col)['count'].cumsum()
        df_plot_norm_cut = df_plot_norm[df_plot_norm['cum_count_norm'] <= cutoff]

        min_val, max_val, rng = find_min_max_rng(df_plot_norm_cut, 'value', False)
        min_val = math.floor(min_val * 10)/10.0
        max_val = math.ceil(max_val * 100)/100.0 + .01

        df_cut = df[(df[plot_col] >= min_val) & (df[plot_col] <= max_val)].reset_index(drop=True)
        s_interval = get_interval(df_cut, plot_col, min_val, max_val, bins=bins)
        df_cut['interval_right'] = [i.right for i in s_interval.values]
        df_cut_plot = df_cut.groupby(l_new_groupby)[plot_col].count().reset_index()
        df_cut_plot.columns = l_groupby_col + ['value', 'count']
        df_cut_plot_norm = (df_cut_plot.groupby(l_groupby_col + ['value'])['count'].apply(lambda x: x.sum()) / \
            df_cut_plot.groupby(l_groupby_col)['count'].apply(lambda x: x.sum())).reset_index()
        return df_cut_plot_norm
    
    def plot_audience_hist_comp(df_plot, l_groupby_col, 
                                title='Audience Comparison Histograms', 
                                filename='audience_comp_hist.html'):
        groupby_str = '-'.join(l_groupby_col)
        if len(l_groupby_col) > 1:
            df_plot['groupby'] = df_plot[l_groupby_col].apply(lambda row: '|'.join(row.values.astype(str)), axis=1)
        else:
            df_plot['groupby'] = df_plot[l_groupby_col]
        uniq_grp = list(set(df_plot['groupby'].values))
        data = [go.Bar(name=grp, 
                       x=df_plot[df_plot[l_groupby_col[0]] == grp]['value'],
                       y=df_plot[df_plot[l_groupby_col[0]] == grp]['count']) for grp in uniq_grp]
        fig = go.Figure(data=data)
        fig.update_layout(
            title=title,
            xaxis_tickfont_size=14,
            yaxis=dict(
                title='Count',
                titlefont_size=16,
                tickfont_size=14,
            ),
            legend=dict(
                x=0,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)'
            ),
            barmode='group',
            bargap=0.15, # gap between bars of adjacent location coordinates.
            bargroupgap=0.1 # gap between bars of the same location coordinate.
        )

        ## NEED TO NORMALISE BY COUNT
        plot(fig, filename=filename)
        
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
