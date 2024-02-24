

def process_metrics(results, current, iterations):
    for key in current.keys():
        old = results[key]
        old.append(current[key]/iterations)
        results[key] = old
        current[key] = 0