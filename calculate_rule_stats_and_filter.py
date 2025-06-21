
### This function can be imported into rule_quality folder and was tested to work with Aerial version 0.3.0
def calculate_rule_stats_and_filter(rules, transactions, max_workers=1, min_support=0.6, min_confidence=0.8):
    """
    Calculate rule quality stats for the given set of rules based on the input transactions.
    """
    if max_workers == 1:
        logger.info("To speed up rule quality calculations, set max_workers > 1 in calculate_rule_stats() "
                    "to process rules in parallel.")
    num_transactions = len(transactions)
    vector_tracker_list = transactions.columns.tolist()

    dataset_coverage = np.zeros(num_transactions, dtype=bool)

    def process_rule(rule, min_support=min_support, min_confidence=min_confidence):
        antecedents_indices = [vector_tracker_list.index(ant) for ant in rule['antecedents']] # feature indices of antecedent
        consequent_index = vector_tracker_list.index(rule['consequent']) # feature indices of the consequent

        # Find transactions where all antecedents are present
        antecedent_matches = np.all(transactions.iloc[:, antecedents_indices] == 1, axis=1)
        consequent_matches = transactions.iloc[:, consequent_index] == 1
        co_occurrence_matches = antecedent_matches & (transactions.iloc[:, consequent_index] == 1)

        antecedents_occurrence_count = np.sum(antecedent_matches)
        consequent_occurrence_count = np.sum(consequent_matches)
        co_occurrence_count = np.sum(co_occurrence_matches)

        support_body = antecedents_occurrence_count / num_transactions if num_transactions else 0
        support_head = consequent_occurrence_count / num_transactions if num_transactions else 0
        rule_support = co_occurrence_count / num_transactions if num_transactions else 0
        rule_confidence = rule_support / support_body if support_body != 0 else 0

        rule['support'] = float(round(rule_support, 3))
        rule['confidence'] = float(round(rule_confidence, 3))
        if rule_support > min_support and rule_confidence > min_confidence:
            rule['zhangs_metric'] = float(round(calculate_zhangs_metric(rule_support, support_body, support_head), 3))
            rule['rule_coverage'] = float(
                round(antecedents_occurrence_count / num_transactions if num_transactions else 0, 3))
            return antecedent_matches, rule
        else:
            return None
    
    # Parallel processing of rules
    results = Parallel(n_jobs=max_workers)(delayed(process_rule)(rule) for rule in rules)
    # ADDED 
    results = [res for res in results if res is not None]
    # Aggregate dataset coverage and collect updated rules
    # By the end, dataset_coverage contains all data points that match at least one rule
    updated_rules = []
    for antecedent_matches, rule in results:
        dataset_coverage |= antecedent_matches
        updated_rules.append(rule)

    if updated_rules == []:
        print('returning none none')
        return None, None
    
    stats = calculate_average_rule_quality(updated_rules)
    stats["data_coverage"] = np.sum(dataset_coverage) / num_transactions

    return {"rule_count": len(updated_rules), "average_rule_coverage": float(round(stats["rule_coverage"], 3)),
            "average_support": float(round(stats['support'], 3)),
            "average_confidence": float(round(stats["confidence"], 3)),
            "data_coverage": float(round(stats["data_coverage"], 3)),
            "average_zhangs_metric": float(round(stats["zhangs_metric"], 3))}, updated_rules
