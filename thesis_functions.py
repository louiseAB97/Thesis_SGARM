from mlxtend.frequent_patterns import fpgrowth, association_rules
import numpy as np 
import pandas as pd
from aerial_plus_folder import aerial_plus
import re
import ast
import time
from aerial import model, rule_extraction, rule_quality
import random
import matplotlib.pyplot as plt
import traceback

def analyze_go_terms_rules(transaction_df_aerial, transaction_df_fp, selected_terms, min_support, min_confidence, row_expansion=24, 
                          aerial_runs=5, n_epochs=3, filter_thresholds=True):
    """
    Analyze GO terms frequecy and rule appearance
    
    Parameters:
    - transaction_df: Input transaction dataframe in which each gene is present multiple times. (because we have multiple samples)
    - selected_terms: List of GO terms with a range of genes that they are attributed to
    - min_support, min_confidence: Thresholds for FP growth AND Aerial to make comparison with FP-growth better
    - row_expansion: Number of samples for term to gene attribution (have to devide by this numner)
    - aerial_runs: Number of times to run Aerial algorithm
    - n_epochs: If row number much bigger than cols then less epochs to prevent overfitting
    """
    
    # dictionary to keep track of counts
    term_counts = {}

    ### Aerial Analysis
    print(f"\nRunning Aerial algorithm {aerial_runs} times...")
    aerial_result, aerial_rules = RunPyAerialnTimes(transaction_df=transaction_df_aerial, selected_terms=selected_terms, ant_sim=min_support, cons_sim=min_confidence,
                                        n=aerial_runs, term_counts=term_counts, row_expansion=row_expansion, n_epochs=n_epochs, filter_thresholds=filter_thresholds)
    try:
        if filter_thresholds == False:
            min_sup_fp = (aerial_result['avg Support'].mean())/2
        else:
            min_sup_fp = min_support
    except: 
        min_sup_fp = min_support
    
    ### FP-Growth Analysis
    time_before= time.time()
    itemsets = fpgrowth(transaction_df_fp, min_support=min_sup_fp, use_colnames=True, max_len=3)
    if itemsets.empty:
        print("No frequent itemsets found with the given min_support.")
        fp_rules = pd.DataFrame()
    else:
        fp_rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
        exec_time = time.time() - time_before
        print(f"FP exec_time: {exec_time}")

        # # filter rules with target features 
        # target_features = {'Gene_Upregulated', 'Gene_Downregulated'}
        # fp_rules_filtered = fp_rules[
        # fp_rules['consequents'].apply(lambda x: bool(x.intersection(target_features)))]
        # print(f'fp rules before filtering: {len(fp_rules)}, after filtering: {len(fp_rules_filtered)}')
        AddTermCountsFromRules(selected_terms=selected_terms, algorithm='FP-Growth', rules=fp_rules, term_counts=term_counts,
                                row_expansion=row_expansion, transaction_df=transaction_df_fp, aerial_runs=aerial_runs)
    
    print("\nGenerating plots...")
    terms = list(term_counts.keys())
    gene_counts = [term_counts[t]['genes_count'] for t in terms]
    fp_rule_counts = [term_counts[t]['fp_rules_count'] for t in terms]
    aerial_rule_counts = [term_counts[t]['aerial_rules_counts'] for t in terms] 
    
    # Sort by gene counts
    sorted_idx = np.argsort(gene_counts)
    terms_sorted = [terms[i] for i in sorted_idx]
    gene_counts_sorted = [gene_counts[i] for i in sorted_idx]
    fp_rule_counts_sorted = [fp_rule_counts[i] for i in sorted_idx]
    if aerial_rule_counts:
        aerial_rule_counts_sorted = [aerial_rule_counts[i] for i in sorted_idx]
    
    # Create plot
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    x = np.arange(len(terms_sorted))
    width = 0.4

    # Gene counts bar chart
    bars = ax1.bar(x, gene_counts_sorted, width, label='Number of genes', color='cornflowerblue')
    ax1.set_xlabel('GO Terms', fontsize=16)
    ax1.set_ylabel('Number of genes attributed to term', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(terms_sorted, rotation=45, ha='right', fontsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.legend(loc='upper left', fontsize=14)

    # Rules line plots
    ax2 = ax1.twinx()
    ax2.plot(x, fp_rule_counts_sorted, 'o-', color='darkorange', label='FP-Growth rules', linewidth=2)
    if aerial_rule_counts:
        ax2.plot(x, aerial_rule_counts_sorted, 's-', color='red', label=f'Aerial rules (avg of {aerial_runs} runs)', linewidth=2)
    ax2.set_ylabel('Number of rules term appears in', fontsize=16)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.legend(loc='upper right', fontsize=14)

    plt.title('GO-term gene attribution vs rule appearances (FP-Growth vs Aerial)', fontsize=18)
    plt.tight_layout()
    # Return results
    results = {
        'fp_rules': fp_rules,
        'term_counts': term_counts,
        'aerial_results': aerial_rules
    }
    
    return results

def AddTermCountsFromRules(selected_terms, algorithm, rules, transaction_df, term_counts, row_expansion, aerial_runs):
    '''
    algorithm must be either 'FP-growth' or 'Aerial' 
    row_expansion refers to how many times each row is expanded (for instance when gene as row design and multiple samples)
    '''
    very_small_nr = 0.00000000000001
    if algorithm == 'FP-Growth':
    # Count GO term occurrences
        for term in selected_terms:
            count_in_rules = rules['antecedents'].apply(lambda x: term in x).sum()
            gene_count = transaction_df[term].sum()
            if not np.isscalar(gene_count):
                gene_count = gene_count.iloc[0]

            if term not in term_counts:
                term_counts[term] = {}

            term_counts[term]['genes_count'] = int(gene_count) // row_expansion
            term_counts[term]['fp_rules_count'] = int(count_in_rules)
            # Ensure the 'aerial_rules_counts' key exists
            term_counts[term].setdefault('aerial_rules_counts', 0)

    if algorithm == 'Aerial':
        # Add aerial term counts to term_counts, divided by number of Aerial runs
        for term in selected_terms:
            # Ensure term is initialized in the dictionary
            if term not in term_counts:
                term_counts[term] = {}
            # Ensure the 'aerial_rules_counts' key exists
            term_counts[term].setdefault('aerial_rules_counts', 0)


            count = 0
            if rules and len(rules) > 0:
                for rule in rules:
                    # Check if term appears in antecedents or consequent
                    if 'antecedents' in rule and term in str(rule['antecedents']):
                        count += 1
                    elif 'consequent' in rule and term in str(rule['consequent']):
                        count += 1

            # Add normalized count
            term_counts[term]['aerial_rules_counts'] += round((count + very_small_nr) / aerial_runs)

def RunPyAerialnTimes(transaction_df, ant_sim, cons_sim, batch_size=2, n_epochs=5, n=5, selected_terms=None, term_counts=None, row_expansion=1, filter_thresholds=True, target_classes=False, 
                      features_of_interest_onehot=False, discard_feature_value=None):
    """
    Run Aerial n times 
    Returns: Results table with averages over runs, and updated rules
    Parameters:
    - transaction_df: Input transaction dataframe 
    - selected_terms: OPTIONAL List of semantic terms
    - ant_sim,  conf_sim: similarity thresholds for aerial
    - aerial_runs: Number of times to run Aerial algorithm
    - batch_size:  default 2, or in case of row_expansion, row_expansion*2
    - filter_threshold: boolean to indicate whether ant_sim and cons_sim should also be treated as minimal support and confidence thresholds to filter rules
    - row_expansion: default 1, except when working with a vertical expanded dataframe with genes as rows and multiple samples. in that case nr of samples.
    - n_epochs: default 5, if row number much bigger than cols then less epochs to prevent overfitting
    - features_of_interest_onehot: if the columns are onehot and you only want rules with the value being
    - discard_feature_value: the value in columns you want to discard from feature value vector , e.g. 'NoChange'
    - target_classes: bit hard coded and used in gene as rows design, can be boolean or list of column names
    optional: row expansion needed when working with the vertically expanded dataframe 
    optional: term_counts and selected_terms, to count term occurrences in generated rules 

    """
    
    # batch_size = row_expansion*2  used for the go terms graph

    runtimes = []
    confidences = []
    supports =[]
    coverages = []
    zhangs = []

    # Track metrics and term occurrences
    results = pd.DataFrame()
    for i in range(n):
        try:
            start_time = time.time()
            from aerial import model

            trained_autoencoder = model.train(transaction_df, epochs=n_epochs, batch_size=batch_size)
            
            features_of_interest = None
            
            if features_of_interest_onehot == True:
                features_of_interest = [{column_name.split("__")[0]: str(1)} for column_name in trained_autoencoder.feature_values
                                if column_name and not column_name.endswith("__0")]
                # # filter the the target classes in case of onehot # ADDED LATER BUT DIDNT WORK WELL ICM ABOVE 
                # if isinstance(target_classes, list):
                #     target_classes = [
                #         feature for feature in features_of_interest
                #         if any(key in target_classes for key in feature)
                #     ] 
                if target_classes == True:
                    target_classes = features_of_interest[0:2] #these are the up and downregulated columns
                else: 
                    target_classes=features_of_interest

            if discard_feature_value != None:
                features_of_interest = [
                    {column_name.split("__")[0]: "1"}
                    for column_name in trained_autoencoder.feature_values
                    if column_name and not column_name.endswith(f"__{discard_feature_value}")]

            rules = rule_extraction.generate_rules(trained_autoencoder, ant_similarity=ant_sim, cons_similarity=cons_sim, max_antecedents=2,
                                                    features_of_interest=features_of_interest, target_classes=target_classes) #added later to comply with new aerial and not inclue __00 z
            
            exec_time = time.time() - start_time
            if len(rules) > 0:
                # the post mining filter on min_sup and min con  doesnt work anymore because i changed it in previous version aerial sourcecode
                print(f"number of rules before filtering {len(rules)}")
                if filter_thresholds == False:
                    summary_stats, updated_rules = rule_quality.calculate_rule_stats(rules, transactions=trained_autoencoder.input_vectors, max_workers=4)
                else:
                    summary_stats, updated_rules = rule_quality.calculate_rule_stats_and_filter(rules, transactions=trained_autoencoder.input_vectors, max_workers=4, min_support=ant_sim, min_confidence=cons_sim)
                if updated_rules != None:
                    print(f"number of rules after filtering {len(updated_rules)}")

                    confidences.append(summary_stats['data_coverage'])
                    supports.append(summary_stats['average_support'])
                    zhangs.append(summary_stats['average_zhangs_metric'])
                    coverages.append(summary_stats['average_confidence'])
                    runtimes.append(exec_time) 
                    if term_counts != None:
                        AddTermCountsFromRules(selected_terms=selected_terms, rules=updated_rules,transaction_df=transaction_df,
                                            term_counts=term_counts, algorithm='Aerial', row_expansion=row_expansion, aerial_runs=n)                            
                    # Store run metrics
                    results.loc[f'Run {i+1}', 'Exec Time (s)'] = exec_time
                    results.loc[f'Run {i+1}', 'Nr Rules'] = len(updated_rules)
                    results.loc[f'Run {i+1}', 'avg Confidence'] = summary_stats['average_confidence']
                    results.loc[f'Run {i+1}', 'avg Support'] = np.round(summary_stats['average_support'],2)
                    results.loc[f'Run {i+1}', 'avg Coverage'] = summary_stats['data_coverage']
                    results.loc[f'Run {i+1}', 'avg Zhangs Metric'] = summary_stats['average_zhangs_metric']
                    print(results)
                    #return results, updated_rules
                #else:
                    #print(f"nr of aerial rules after filtering for minimal confidence and support:0")
                    #return [], []
        except Exception as e:
                print(f"Run {i+1} failed: {e}")
                traceback.print_exc()
    
    # Check if the variable has been defined
    if 'updated_rules' not in locals() or updated_rules is None:
        updated_rules = {}

    return results, updated_rules



def RunPipelineWithPyAerial(transaction_df, metadata_df=None, target_column=None, target_pattern=None, max_antecedents=2
                            ,one_hot_cols=None , cols_interest=None, min_sup=0.6, min_conf=0.8):
    """
    transaction_df can be any kind of df where subjects are in the rows (can be (fold-)change already)
    metadata has to have the same subject ids in the rows if one is change than metadata must be change to
    """
    if metadata_df is not None and not metadata_df.empty:
        transaction_df = transaction_df.join(metadata_df)


    # Remove columns from transaction_df where there is only 1 value (non-informative)
    constant_cols = transaction_df.columns[transaction_df.nunique(dropna=False) <= 1].tolist()
    if constant_cols:
        print("Dropped constant columns:", constant_cols)

    # Drop
    transaction_df_use = transaction_df.loc[:, transaction_df.nunique(dropna=False) > 1]
    print(f"nr of columns left:{len(transaction_df_use.columns)} ")

    #rules, stats = PyAerial_pipeline(transaction_df=transaction_df_use, cols_interest=cols_interest, 
    #                                 one_hot_cols=one_hot_cols, min_sup=min_sup, min_conf=min_conf)
    #return(rules, stats)


def GetChangeDfs(transaction_df_paired, metadata):
    """
    Computes difference between condition A and B for each subject based on super_id rows (e.g., 123A, 123B).
    Returns a DataFrame: rows = subject IDs, columns = gene/pathway features.
    It should have less rows in the end
    """
    df = transaction_df_paired.copy()

    # Get subject ID (e.g., 123) by stripping last char
    df['subject_id'] = df.index.str[:-1]
    df['condition'] = df.index.str[-1]

    # Separate condition A and B
    df_A = df[df['condition'] == 'A'].drop(columns=['condition'])
    df_B = df[df['condition'] == 'B'].drop(columns=['condition'])

    # Set subject_id as index so we can align A and B
    df_A = df_A.set_index('subject_id')
    df_B = df_B.set_index('subject_id')

    # Make sure both A and B are available for all subjects
    common_subjects = df_A.index.intersection(df_B.index)
    df_A = df_A.loc[common_subjects]
    df_B = df_B.loc[common_subjects]

    # Calculate change before and after treatment
    df_change = df_B - df_A
    df_change = df_change.round(2)

    # preprocess metadata (metadata is indexed with sample_id but there is a column super_id that is the same as the initial index of ssGSEA_matrix)
    # we need 1 row per subject id, most column values for the same subject_id are the same; in this case drop column
    # in case the values are different (e.g. body weight, fat percentage etc) perform B - A and call the column change_colname
    metadata = metadata.copy()
    metadata['subject_id'] = metadata['super_id'].str[:-1]
    metadata['condition'] = metadata['super_id'].str[-1]

    meta_A = metadata[metadata['condition'] == 'A'].set_index('subject_id')
    meta_B = metadata[metadata['condition'] == 'B'].set_index('subject_id')

    common_meta_subjects = meta_A.index.intersection(meta_B.index)
    meta_A = meta_A.loc[common_meta_subjects]
    meta_B = meta_B.loc[common_meta_subjects]

    # Remove non-informative columns
    meta_change = pd.DataFrame(index=common_meta_subjects)
    for col in meta_A.columns:
        if col in ['super_id', 'condition', 'timepoint', 'timepoint_letter']:
            continue
        if (meta_A[col] != meta_B[col]).all():
            try:
                diff = (meta_B[col].astype(float) - meta_A[col].astype(float)).round(2)
                meta_change[f'change_{col}'] = diff
            except:
                continue  # skip non-numeric columns that differ
        else:
            meta_change[col] = meta_A[col] #keeps value that it the same for both timepoints e.g. treatment

    return df_change, meta_change


def discretize_features_minmax_bins(dataset, bins=5, global_minmax=True ):
    '''
    This code divides the data into n bins based on min and max values. 
    If global_minmax=true it uses the global values, if False it takes the minmax of each column
    '''
    X = dataset.copy()
    
    # Select only numeric columns
    numerical_columns = X.select_dtypes(include=[np.number]).columns
    
    if global_minmax: # Get the overall min and max values from the entire dataset (since all columns are chaneg in ssGSEA this is good -they are in the same scale)

        min_value = X[numerical_columns].min().min()
        max_value = X[numerical_columns].max().max()
        
        # Create bins based on the global min and max values
        bin_edges = np.linspace(min_value, max_value, bins + 1)
        discretized_columns = {
            col: pd.cut(X[col], bins=bin_edges, include_lowest=True)
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", "-", regex=False)
            for col in numerical_columns
        }
    else: # devide over equal sized buckets (Better for metadata in which columns are all different scale)
        discretized_columns = {
            col: pd.qcut(X[col], q=bins, precision=1)
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", "-", regex=False)
            for col in numerical_columns
        }    
    # Discretize each numeric column using equal-width binning based on the global range
    return pd.DataFrame(discretized_columns)

def discretize_features_by_abs_threshold(dataset, bins=3, threshold=1.0):
    '''
    This code divides the data into n bins based on absolute threshold. 
    When above or under threshold: 'Up', 'Down', otherwise 'NoChange'
    '''
    X = dataset.copy()
    discretized_columns = {}

    for col in X.columns:
        vals = X[col]
        discretized = vals.apply(lambda x: 'Up' if x >= threshold else ('Down' if x <= -threshold else 'NoChange'))
        discretized_columns[col] = discretized

    return pd.DataFrame(discretized_columns, index=X.index)

### Data loading functions
def parse_frozenset(text):
    '''
    how to use: 
    frequent_itemset['itemsets'] = frequent_itemset['itemsets'].apply(parse_frozenset)
    '''
    # Extract the content inside the frozenset
    match = re.search(r"frozenset\({(.*)}\)", text)
    if match:
        content = match.group(1)
        # Parse the content (handle quotes appropriately)
        return frozenset(ast.literal_eval('{' + content + '}'))
    return frozenset()

def GetFoldChangeDf(raw_expr_mat, silent=True, subjects_as_rows=False):
    '''
    Takes as input a raw expression matrix in which genes are rows and colums 
    are in the structure of c(sample_nr, letter) in which letter is condition. example 1A, 1B would besample 1 conditions A and B

    Returns a df in which there is a foldchange between the two conditions for each sample instead of two different columns for each sample. 
    if subject as rows is true, the columns contain fold changes for genes for each sample.
    '''

    df = raw_expr_mat
    df_fc = pd.DataFrame()
    subject_columns = [col for col in df.columns if col[0].isdigit()]
    subject_columns = [col[:-1] for col in subject_columns]

    for subject in subject_columns:
    # Assuming condition A and B are in columns like 'subjectA' and 'subjectB'
    # Adjust these condition names accordingly
    # Calculate the fold change
        if f"{subject}A" in df.columns and f"{subject}B" in df.columns:
            # Calculate the fold change if both condition columns exist
            if subjects_as_rows == True:
                df_fc[f"{subject}"] = np.log2(df[f"{subject}B"] / df[f"{subject}A"])
            else: 
                df_fc[f"{subject}_FC"] = np.log2(df[f"{subject}B"] / df[f"{subject}A"])
        else:
            # If either condition column doesn't exist, skip this subject
            if silent == False:
                print(f"Skipping {subject}: One or both condition columns ('{subject}A' or '{subject}B') are missing.")

    if subjects_as_rows == True: 
        return(df_fc.T)
    else:
        return(df_fc)


def MakeOneHotDf(df_fc, threshold, expand_columns=True):
    '''
    Takes the fold change df and makes it into a one-hot or categorical df.
    Threshold indicates at which log fold change we consider the change meaningful.
    '''
    if expand_columns:
        df_fc_t = df_fc
        df_one_hot = pd.concat([
            (df_fc_t > threshold).astype(int).add_suffix("_upregulated"),
            (df_fc_t < -threshold).astype(int).add_suffix("_downregulated")
        ], axis=1)
        df_one_hot = df_one_hot.reindex(sorted(df_one_hot.columns), axis=1)
        df_one_hot.index.name = "Subject"
        #before without astype(str)
        df_one_hot.index = df_one_hot.index.astype(str).str.replace("_FC", "", regex=False)
        df_one_hot_b = df_one_hot.astype("int")
        return df_one_hot_b
    else: # for aerial it not needed to work with a onehot df, just threshold. 
        def encode(val):
            if val > threshold:
                return 'upregulated'
            elif val < -threshold:
                return 'downregulated'
            else:
                return 'nochange'

        df_categorical = df_fc.map(encode)
        df_categorical.index.name = "Subject"
        df_categorical.index = df_categorical.index.str.replace("_FC", "", regex=False)
        return df_categorical



### Functions with genes as rows 

def RunPipelineWithFPGrowth(expr_matrix_raw, geneset_name_for_save, save_folder='Benchmarking', logfc_threshold=0.2,
                             min_support=0.6, max_len=5, metric='confidence', min_threshold_metric=0.7, gene_names_to_include=[]):
    expr_matrix = expr_matrix_raw
    
    if gene_names_to_include != []:
        expr_matrix = expr_matrix.loc[expr_matrix.index.intersection(gene_names_to_include)]

    fc_df = GetFoldChangeDf(expr_matrix)
    onehot_df = MakeOneHotDf(fc_df, threshold=logfc_threshold)
    
    freq_itemset = fpgrowth(onehot_df, min_support=min_support, use_colnames=True, verbose=False, max_len=max_len)
    if not freq_itemset.empty:  
        freq_itemset.to_csv(f"{save_folder}/fp_growth_{geneset_name_for_save}_genes_minSupport_{min_support}_logFCthreshold_{logfc_threshold}_maxLen_{max_len}_freqItemset.csv", index=False)
        rules = association_rules(freq_itemset, metric=metric, min_threshold=min_threshold_metric)

        if not rules.empty:
            rules.to_csv(f"{save_folder}/fp_growth_sig_genes_minSupport_{min_support}_logFCthreshold_{logfc_threshold}_maxLen_{max_len}_rules.csv")
            avg_conf=rules['confidence'].mean()
            avg_sup =rules['support'].mean()
            avg_lift=rules['lift'].mean()
            print(f"Avg confidence: {avg_conf:.3f}, Avg support: {avg_sup:.3f}, Avg lift: {avg_lift:.3f}")
            return (rules)
        else:
            print('No rules found, consider lowering logfc_threshold')
    else: 
        print('No frequent itemsets and rules found, consider lowering logfc_threshold')
        return
    freq_itemset = fpgrowth(onehot_df, min_support=min_support, use_colnames=True, verbose=False, max_len=max_len)

def RunPipelineWithAerial(fc_df, sem_df=None, stop_after_minutes=30, nr_of_genes_to_include='All'):
    
    #OLD aeerial
     # Create a copy to avoid SettingWithCopyWarning
    fc_df_copy = fc_df.copy()
    
    # Limit genes if specified
    if nr_of_genes_to_include != 'All':
        try:
            fc_df_copy = fc_df_copy.iloc[:nr_of_genes_to_include]
        except:
            print("nr_of_genes_to_include must be int")
    
    if sem_df != None:
        # Filter semantic DataFrame to match fold-change DataFrame indices
        sem_df_copy = sem_df.loc[list(fc_df_copy.index), :].copy()
        
        # Drop columns that contain only zeros
        sem_df_copy = sem_df_copy.loc[:, (sem_df_copy != 0).any(axis=0)]
        
        # Properly convert all column values 
        sem_df_processed = sem_df_copy.astype(str)
        fc_df_processed = fc_df_copy.astype(float)

        transaction_df = fc_df_processed.join(sem_df_processed)
    else: 
        transaction_df = fc_df_copy
    
    print(f'The shape of the created transaction df is {transaction_df.shape}')
    

    Aerial = aerial_plus.AerialPlus()

    # Step 1
    print("Creating input vectors, this might take some time")
    Aerial.create_input_vectors(transaction_df)
    
    # Step 2
    print("Training, please wait")
    Aerial.train()

    # Step 3
    print("Generating association rules, this might take even longer")
    rules, exec_time = Aerial.generate_rules()        

    # Now lets see what the stats functions do
    result = Aerial.calculate_stats(rules, transaction_df, exec_time)
    updated_rules = result[1]
    exec_time = result[0][1]
    print(f" Nr of rules after updating {result[0][0]}, exec_time: {result[0][1]}, support:{result[0][2]}, confidence: {result[0][3]}, coverage: {result[0][4]}")

    return([Aerial, rules, updated_rules])


def filter_rules(rules_list, antecedent_pattern='__1', consequent_pattern='__0', 
                 min_support=0.6, min_confidence=0.0):

    filtered_rules = []
    
    for rule in rules_list:
        # Check support and confidence thresholds
        if rule['support'] < min_support or rule['confidence'] < min_confidence:
            continue
            
        # Check consequent pattern if specified
        if consequent_pattern is not None and consequent_pattern not in rule['consequent']:
            continue
            
        # Check antecedent patterns if specified
        if antecedent_pattern is not None:
            antecedents = rule['antecedents']
            if not all(antecedent_pattern in ant for ant in antecedents):
                continue
                
        filtered_rules.append(rule)
    
    print(f"Filtered from {len(rules_list)} to {len(filtered_rules)} rules")
    return filtered_rules



def randomly_downsize(df, row_frac=0.5, col_frac=0.5, random_state=42):
    
    np.random.seed(random_state)
    
    # Randomly select rows and columns
    n_rows = int(len(df) * row_frac)
    n_cols = int(len(df.columns) * col_frac)
    
    selected_rows = np.random.choice(df.index, n_rows, replace=False)
    selected_cols = np.random.choice(df.columns, n_cols, replace=False)
    
    return df.loc[selected_rows, selected_cols]

def select_top_signal_columns(df, col_frac=0.5, method="signal"):
    """
    Selects the top fraction of columns based on information content:
    - 'signal': columns with most 1s (works for binary data)
    - 'diversity': columns with highest number of unique values

    Parameters:
        df (pd.DataFrame): Input DataFrame
        col_frac (float): Fraction of columns to keep
        method (str): 'signal' or 'diversity'

    Returns:
        pd.DataFrame: Subset with selected informative columns
    """
    if not (0 < col_frac <= 1):
        raise ValueError("col_frac must be between 0 and 1.")

    n_cols = int(len(df.columns) * col_frac)

    if method == "signal":
        # Count number of 1s in each column (assumes binary)
        col_scores = df.sum()
    elif method == "diversity":
        # Count number of unique values in each column
        col_scores = df.nunique()
    else:
        raise ValueError("method must be 'signal' or 'diversity'.")

    # Select top columns
    top_cols = col_scores.sort_values(ascending=False).head(n_cols).index
    return df.loc[:, top_cols]

### functions with subjects as rows