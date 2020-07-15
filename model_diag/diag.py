"""
Model diagnosis module
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from dateutil.rrule import rrule, DAILY

def compare_distr(l_values, l_label, num_bins=100, xrange=[0, 1], norm_hist=True):
    for values, label in zip(l_values, l_label):
        sns.distplot(values, hist=False, label=label, bins=num_bins, norm_hist=norm_hist)
    plt.xlim(xrange[0], xrange[1])
    plt.show()

def compare_feature_distr(base_df, new_df, features, ks_threshold=0.05, ignore_zeros_nulls=False):
    # compare distributions of features across two data frames
    # can leverage psi to quickly find out which feature are very different, two groups only
    l_p_values = []
    for feature in features:
        base_feat_values = base_df[feature].values
        new_feat_values = new_df[feature].values
        ks_stat, p_value = stats.ks_2samp(base_feat_values, new_feat_values)
        l_p_values.append(p_value)
        if p_value < ks_threshold:
            print(f'The feature {feature} exhibits different distributions between data sets')
    df = pd.DataFrame('feature': features, 'ks_p_value': l_p_values)
    return df

def group_by_metrics():
    # groupby and compare metrics
    pass

def get_counts_proportions(scores, num_bins=10):
    """
    Get percentile of descriptions
    :param scores: scores
    :param num_bins: number of bins for histogram definition

    :return: Array of proportions
    """
    try:
        bins_range = np.linspace(0, 1, num_bins + 1)
        counts = np.histogram(scores, bins=bins_range)[0]
        props = counts / counts.sum()
        return props
    except Exception as e:
        raise

def get_psi(base_scores, new_scores, num_bins=10, return_df=False):
    """
    Get population stability index
    :param base_scores: base scores (original scores from original model fit)
    :param new_scores: model scores to compare baseline to
    :param num_bins: number of bins for histogram definition
    :param return_df: Boolean to return dataframe

    :return psi: population stability index (float)
    :return dataframe: pandas dataframe
    """
    try:
        prop_base = get_counts_proportions(base_scores, num_bins=num_bins)
        prop_new  = get_counts_proportions(new_scores, num_bins=num_bins)
        prop_diff = prop_base - prop_new
        log_prop_diff = np.log(prop_base/prop_new)
        df = pd.DataFrame({'base_prop': prop_base,
                           'new_prop': prop_new, 
                           'diff': prop_diff,
                           'log_diff': log_prop_diff, 
                           'log_diff': log_prop_diff
                          })
        if any(np.isinf(prop_base/prop_new)) or any(np.isinf(prop_new/prop_base)):
            self.logger.warning('There are bins where counts exist in one dataset but not the other.')

        log_prop_diff[np.isnan(log_prop_diff) | np.isinf(log_prop_diff)] = 0.
        df['log_diff_adj'] = log_prop_diff
        df['psi_index'] = prop_diff * log_prop_diff
        psi = df['psi_index'].sum()
        if return_df:
            return psi, df
        else: 
            return psi

    except Exception as e:
        raise


def chk_model_psi(base_scores, new_scores, cutoff=0.1):
    """
    Check that population stability index is below a certain threshold
    :param base_scores: base scores (original scores from original model fit)
    :param new_scores: model scores to compare baseline to
    :param logger: python logger
    :param cutoff: cutoff

    :return: None
    """
    try:
        psi_metric = get_psi(base_scores, new_scores)
        if psi_metric > cutoff:
            logger.warning(f'Population stability index has exceeded cutoff ({cutoff:0.2f}): {psi_metric:0.3f}')
        else:
            logger.info(f'Population stability index: {psi_metric:0.3f}')
    except Exception as e:
        raise


class DiagModule:
    """
    Class for running tests on dataframes for churn
    model evaluation
    """
    def __init__(self, df, logger):
        """
        Initialise
        :param df: pandas dataframe for testing
        :param logger: python logger
        """
        try:
            self.df     = df
            self.logger = logger
        except Exception as e:
            raise

    def __get_counts_proportions(self, scores, num_bins=10):
        """
        Get percentile of descriptions
        :param scores: scores
        :param num_bins: number of bins for histogram definition

        :return: Array of proportions
        """
        try:
            bins_range = np.linspace(0, 1, num_bins + 1)
            counts = np.histogram(scores, bins=bins_range)[0]
            props = counts / counts.sum()
            return props
        except Exception as e:
            raise

    def __get_psi(self, base_scores, new_scores, num_bins=10, return_df=False):
        """
        Get population stability index
        :param base_scores: base scores (original scores from original model fit)
        :param new_scores: model scores to compare baseline to
        :param num_bins: number of bins for histogram definition
        :param return_df: Boolean to return dataframe

        :return psi: population stability index (float)
        :return dataframe: pandas dataframe
        """
        try:
            prop_base = self.__get_counts_proportions(base_scores, num_bins=num_bins)
            prop_new  = self.__get_counts_proportions(new_scores, num_bins=num_bins)
            prop_diff = prop_base - prop_new
            log_prop_diff = np.log(prop_base/prop_new)
            df = pd.DataFrame({'base_prop': prop_base,
                               'new_prop': prop_new, 
                               'diff': prop_diff,
                               'log_diff': log_prop_diff, 
                               'log_diff': log_prop_diff
                              })
            if any(np.isinf(prop_base/prop_new)) or any(np.isinf(prop_new/prop_base)):
                self.logger.warning('There are bins where counts exist in one dataset but not the other.')

            log_prop_diff[np.isnan(log_prop_diff) | np.isinf(log_prop_diff)] = 0.
            df['log_diff_adj'] = log_prop_diff
            df['psi_index'] = prop_diff * log_prop_diff
            psi = df['psi_index'].sum()
            if return_df:
                return psi, df
            else: 
                return psi

        except Exception as e:
            raise


    def __chk_model_psi(self, base_scores, new_scores, cutoff=0.1):
        """
        Check that population stability index is below a certain threshold
        :param base_scores: base scores (original scores from original model fit)
        :param new_scores: model scores to compare baseline to
        :param logger: python logger
        :param cutoff: cutoff

        :return: None
        """
        try:
            psi_metric = self.__get_psi(base_scores, new_scores)
            if psi_metric > cutoff:
                self.logger.warning(f'Population stability index has exceeded cutoff ({cutoff:0.2f}): {psi_metric:0.3f}')
            else:
                self.logger.info(f'Population stability index: {psi_metric:0.3f}')
        except Exception as e:
            raise

    def __chk_crn_avail(self, df, crn_list, chk_col):
        """
        Check that all customers in crn_list have a corresponding record in a dataframe
        :param crn_list: list of CRNs that should have a record
        :param chk_col: name column record to check

        :return: None
        """
        
        assert 'crn' in df.columns, "CRN column is available in dataframe"
        assert chk_col in df.columns, f"Column {chk_col} is not available in dataframe"
        try:
            self.logger.info(f'Checking that CRNs are aligned')
            crn_list = pd.DataFrame(list(crn_list),columns=['crn'])
            crn_mrg  = crn_list.merge(df,on='crn',how='left')
            nan_list = crn_mrg.loc[crn_mrg[chk_col].isna(),:].index.tolist()

            # check
            if len(nan_list):
                self.logger.error(f"There are {len(nan_list)}/{len(df)} records in dataframe with {chk_col} NaN")
            else:
                self.logger.info(f"There are no records in dataframe with {chk_col} NaN")
        except Exception as e:
            raise

    def __chk_not_nan(self,chk_col=None):
        """
        Check that given column is not NaN
        :param chk_col: name of column to check, if None passed then check all
        :return: None
        """
        try:
            # check all if no specific column passed
            if chk_col is None:
                if self.df.isna().any().any():
                    self.logger.warning("There are columns containing NaNs records in dataframe")
            else:
                assert chk_col in self.df.columns, f"Column {chk_col} is not available in dataframe"
                n_nan = self.df[chk_col].isna().sum()
                if n_nan:
                    self.logger.error(f"Column {chk_col} contains {n_nan} NaN records")
        except Exception as e:
            raise

    def __chk_date_range(self, date_col, start_dt, end_dt):
        """
        Check that date column is within specified bounds
        :param date_col: name of date column in dataframe
        :param start_dt: start date (expected min date)
        :param end_dt: end date (expected max date)

        :return: None
        """
        assert date_col in self.df.columns, f"Date column {date_col} is not available in dataframe"
        try:
            if self.df[date_col].min()<start_dt:
                self.logger.warning(f"Selected data cohort precedes start_dt supplied: {start_dt}")
            if self.df[date_col].max()>end_dt:
                self.logger.warning(f"Selected data cohort exceeds end_dt supplied: {end_dt}")
        except Exception as e:
            raise

    def __chk_missing_dates(self, date_col, date_list):
        """
        Check that date column is within specified bounds
        :param date_col: name of date column in dataframe
        :param date_list: name of date column in dataframe

        :return: None
        """
        assert date_col in self.df.columns, f"Date column {date_col} is not available in dataframe"
        try:
            missing_dates = set(date_list)-set(self.df[date_col])
            for miss_dt in missing_dates:
                self.logger.error(f"Dataframe {date_col} does not have records for date: {miss_dt}")
        except Exception as e:
            raise
            
    def run_tests(self, test_suite, crn_list=None, base_scores=None, start_dt=None, end_dt=None):
        """
        Run tests suites
        :param test_suite: name of test suite in {'crn', 'trx', 'scores'}
        :param crn_list: pandas series containing CRNs
        :param base_scores: pandas series containing set of model scores from training
        """
        allow_tests = ['crn','trx','scores']
        assert test_suite in allow_tests, f"Test suite must be in one of the following: {allow_tests}"
        assert crn_list is not None or test_suite=='crn', "CRN list must be passed if test_suite =/= crn"
        assert base_scores is not None or test_suite!='scores', "Original model scores must be passed if test_suite = scores"
        assert (start_dt is not None and end_dt is not None) or test_suite!='crn', "Must set start_dt, end_dt for test_suite = crn"
        try:
            self.logger.info(f"Running tests for: {test_suite}")

            if test_suite=='crn':
                self.__chk_not_nan('crn')
                self.__chk_not_nan('first_order_date')
                self.__chk_not_nan('dt_t0')
                self.__chk_date_range('dt_t0', start_dt, end_dt)

                # get list of dates spanning start_dt:end_dt
                l_dates = [d for d in rrule(DAILY, dtstart=start_dt, until=end_dt)]
                self.__chk_missing_dates('dt_t0',l_dates)

            elif test_suite=='trx':
                self.__chk_not_nan('crn')
                self.__chk_not_nan('crn_order_idx')
                self.__chk_crn_avail(self.df.loc[self.df.crn_order_idx==1,:], crn_list, 'crn_order_idx')

            elif test_suite=='scores':
                self.__chk_not_nan()
                self.__chk_crn_avail(self.df, crn_list, 'prob_churn')
                self.__chk_model_psi(base_scores, self.df.prob_churn ,cutoff=0.25)

        except Exception as e:
            raise