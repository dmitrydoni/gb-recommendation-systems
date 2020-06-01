def hit_rate(recommended_list, bought_list):
    """Hit rate"""
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    hit_rate = (flags.sum() > 0) * 1
    
    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    """Hit rate@k"""
    
    bought_list = np.array(bought_list)
    recommended_list_top_k = np.array(recommended_list[:k])
    
    flags = np.isin(bought_list, recommended_list_top_k)
    
    hit_rate_at_k = (flags.sum() > 0) * 1
    
    return hit_rate_at_k


def precision(recommended_list, bought_list):
    """Precision"""
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    """Precision@k"""
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision_at_k = flags.sum() / len(recommended_list)
    
    return precision_at_k


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    """Money Precision@k"""
        
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    
    bought_list = bought_list
    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]
    
    flags = np.isin(recommended_list, bought_list)
    k_ones = np.ones(k)
    
    # Calculate scalar products
    revenue_k_recommended_relevant = np.dot(flags[:k], prices_recommended)
    revenue_k_recommended = np.dot(k_ones, prices_recommended)
    
    money_precision_at_k = revenue_k_recommended_relevant / revenue_k_recommended
    
    return money_precision_at_k


def recall(recommended_list, bought_list):
    """Recall"""
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    """Recall@k"""
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    recall_at_k = flags.sum() / len(bought_list)
    
    return recall_at_k


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    """Money Recall@k"""
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    prices_bought = np.array(prices_bought)
    
    bought_list = bought_list
    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]
    
    flags = np.isin(bought_list, recommended_list)
    relevant_ones = np.ones(len(bought_list))
    
    # Calculate scalar products
    revenue_k_recommended_relevant = np.dot(flags[:k], prices_bought[:k])
    revenue_relevant = np.dot(relevant_ones, prices_bought)
    
    money_recall_at_k = revenue_k_recommended_relevant / revenue_relevant
    
    return money_recall_at_k


def ap_k(recommended_list, bought_list, k=5):
    """Average Precision@k"""
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(1, k+1):
        
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k
            
    result = sum_ / sum(flags)
    
    return result


def map_k(recommended_list, bought_lists, k=5):
    """Mean Average Precision@k"""
    
    ap_k = 0
    apk_list = []  
    
    for bought_list in bought_lists:
        ap_k = ap_k(recommended_list, bought_list, k)
        apk_list.append(ap_k)
    
    map_k = sum(apk_list) / len(apk_list)
    
    return map_k

