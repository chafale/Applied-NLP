"""
@course: CSCI 544 Applied NLP
@hw : HW4
@file : utility file
@author : Ashwin Chafale
@usc-id : 1990624801
"""

import re


def data_preprocessing(reviews):
    # converting all review to lower case
    processed_reviews = [r.lower() for r in reviews]
    # remove punctuation from the reviews
    processed_reviews = [re.sub(r'[^\w\s]', '', pr) for pr in processed_reviews]
    # remove stop words from the reviews
    processed_reviews = stopWord_removal(processed_reviews)
    return processed_reviews


def stopWord_removal(reviews):
    stopWord_list = {'could', 'least', 'mine', 'see', 'put', 'nevertheless', 'whereby', 'sometime', 'through',
                     'neither', 'every', 'thence', 'beyond', 'even', 'yourselves', 'on', 'only', 'many', 'hers', 'fify',
                     'seemed', 'whither', 'thickv', 'below', 'too', 'formerly', 'anyone', 'much', 'nine', 'should',
                     'but', 'some', 'towards', 'other', 'with', 'ourselves', 'else', 'anyway', 'everyone', 'detail',
                     'fill', 'herein', 'it', 'serious', 'thereupon', 'last', 'couldnt', 'thru', 'among', 'everywhere',
                     'their', 'de', 'seems', 'beside', 'between', 'if', 'toward', 'are', 'several', 'whoever', 'he',
                     'three', 'nor', 'bill', 'then', 'very', 'full', 'top', 'never', 'became', 'cant', 'without',
                     'these', 'since', 'etc', 'off', 'me', 'whereafter', 'somehow', 'somewhere', 'anyhow', 'out', 'own',
                     'after', 'ltd', 'into', 'she', 'no', 'whence', 'at', 'wherein', 'might', 'get', 'around', 'during',
                     'already', 'we', 'per', 'same', 'whereupon', 'become', 'front', 'hence', 'again', 'amoungst', 'of',
                     'while', 'yet', 'because', 'side', 'across', 'anywhere', 'something', 'eg', 'amount', 'enough',
                     'the', 'was', 'find', 'were', 'nothing', 'give', 'sometimes', 'how', 'more', 'thereby', 'empty',
                     'before', 'bottom', 'show', 'for', 'about', 'however', 'done', 'from', 'once', 'our', 'you', 'one',
                     'take', 'fifteen', 'itself', 'this', 'moreover', 'thin', 'both', 'amongst', 'sincere', 'cry',
                     'made', 'will', 'latter', 'by', 'hereupon', 'in', 'behind', 'whose', 'co', 'whether', 'con', 'ie',
                     'up', 'why', 'elsewhere', 'above', 'until', 'otherwise', 'that', 'namely', 'noone', 'four',
                     'therein', 'beforehand', 'ever', 'therefore', 'her', 'which', 'than', 'do', 'may', 'yours', 'less',
                     'though', 'against', 'first', 'a', 'there', 'system', 'been', 'and', 'to', 'whenever', 'is',
                     'back', 'eleven', 'they', 'any', 'twenty', 'being', 'us', 'whole', 'throughout', 'often', 'now',
                     'perhaps', 'sixty', 'herself', 'twelve', 'third', 'move', 'those', 'ours', 'not', 'mill', 'former',
                     'what', 'himself', 'describe', 'seem', 'still', 'all', 'together', 'seeming', 'indeed', 'found',
                     'had', 'keep', 'its', 'next', 'his', 'whom', 'most', 'hasnt', 'further', 'over', 'mostly', 'so',
                     'becomes', 'others', 'forty', 'few', 'hereafter', 'un', 'under', 'meanwhile', 'please', 'go',
                     'call', 'within', 'has', 'five', 'as', 'although', 'besides', 'down', 'alone', 'also', 'name',
                     'have', 'six', 'someone', 'fire', 'becoming', 'along', 'nobody', 'hereby', 'always', 'well',
                     'anything', 'when', 'two', 'upon', 'afterwards', 'them', 'be', 'inc', 'am', 'can', 'my', 'whereas',
                     'latterly', 'rather', 'either', 'must', 'almost', 'interest', 'thus', 'everything', 'wherever',
                     'part', 'none', 'ten', 'an', 'thereafter', 'would', 'nowhere', 'eight', 'another', 'cannot',
                     'onto', 'your', 'hundred', 'whatever'}

    processed_reviews = []
    for review in reviews:
        temp = " ".join([" ".join([r for r in review.split() if r not in stopWord_list])])
        processed_reviews.append(temp)
    return processed_reviews
