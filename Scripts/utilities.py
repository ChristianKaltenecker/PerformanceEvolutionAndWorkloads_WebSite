import math
import scipy.stats

def confidence_interval_disjoint(case_study, first_mean, first_sd, second_mean, second_sd, confidence=0.95):
    repetitions_3 = ['OpenVPN', 'Opus', 'MongoDB']
    # Each configuration of each case study was repeated either 3 or 5 times
    if case_study in repetitions_3:
        n = 3
    else:
        n = 5
    first_se = first_sd * math.sqrt(n)
    second_se = second_sd * math.sqrt(n)
    first_h = first_se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    second_h = second_se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

    return first_mean + first_h < second_mean - second_h or first_mean - first_h > second_mean + second_h

def write_frequency(output_file, revision_frequency):
    revision_frequency_ranking = list(revision_frequency.keys())
    revision_frequency_ranking = list(filter(lambda z: revision_frequency[z] != 0, revision_frequency_ranking))
    revision_frequency_ranking.sort(key=lambda z: revision_frequency[z], reverse=True)
    for revision in revision_frequency_ranking:
        output_file.write(revision + ": " + "{0:.2f}".format(revision_frequency[revision]) + "; ")
