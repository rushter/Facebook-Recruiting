# coding:utf-8
import numpy as np

categories = ['auto parts', 'books and music', 'clothing', 'computers',
              'furniture', 'home goods', 'jewelry', 'mobile',
              'office equipment', 'sporting goods']

countries_list = ['us', 'in', 'py', 'ru', 'th', 'id', 'za', 'ng', 'sd',
                  'au', 'hr',
                  'np', 'iq', 'bd', 'tr', 'ch', 'ke', 'uk', 'fr', 'pk', 'my',
                  'vn',
                  'ro', 'gh', 'ua', 'pl', 'by', 'ar', 'zm', 'lk', 'ph', 'br',
                  'es',
                  'mx', 'il', 'qa', 'nl', 've', 'sg', 'gt', 'ae', 'az', 'uz',
                  'ht',
                  'tz', 'gm', 'dk', 'no', 'kw', 'mk', 'hu', 'it', 'ml', 'sv',
                  'bn',
                  'ni', 'cn', 'et', 'ge', 'mw', 'ee', 'ye', 'kr', 'tn', 'gr',
                  'at',
                  'cm', 'ca', 'mn', 'rs', 'sz', 'pe', 'jp', 'sl', 'bh', 'zw',
                  'bg',
                  'de', 'eu', 'cr', 'jo', 'ie', 'sa', 'eg', 'dz', 'hk', 'ec',
                  'si',
                  'lv', 'na', 'nan', 'mt', 'ug', 'kg', 'se', 'bb', 'sc', 'sn',
                  'om',
                  'fi', 'cl', 'ma', 'am', 'lr', 'be', 'bf', 'kh', 'md', 'ly',
                  'al',
                  'ba', 'bo', 'lt', 'ga', 'mr', 'jm', 'bj', 'mu', 'pa', 'cz',
                  'ao',
                  'lu', 'me', 'af', 'kz', 'hn', 'ls', 'uy', 'lb', 'cy', 'sk',
                  'ir',
                  'la', 'dj', 'bz', 'ci', 'is', 'mg', 'so', 'co', 'pt', 'gy',
                  'td',
                  'rw', 'pr', 'bw', 'gq', 'cv', 'mc', 'ne', 'tg', 'bi', 'sy',
                  'tt',
                  'cd', 'sb', 'mz', 'mm', 'tj', 'tw', 'gu', 'cg', 'gl', 'nz',
                  'mv',
                  'ps', 'tm', 'ag', 'ad', 'sr', 'ws', 'je', 'do', 'li', 'fj',
                  'nc',
                  'gi', 'cf', 'mo', 'dm', 'bt', 're', 'fo', 'mp', 'bm', 'gn',
                  'tl',
                  'pg', 'pf', 'vc', 'zz', 'bs', 'aw', 'gb', 'vi', 'mh', 'tc',
                  'an',
                  'er', 'gp']


def feature_importance(df, model, f_threshhold=10):
    f_importance = model.feature_importances_
    features_list = df.columns.values
    f_importance = 100.0 * (f_importance / f_importance.max())
    important_idx = np.where(f_importance > f_threshhold)[0]
    important_features = features_list[important_idx]
    sorted_idx = np.argsort(f_importance[important_idx])[::-1]
    print "\nFeatures importance:\n", list(important_features[sorted_idx])


