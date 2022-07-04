import re,os

def prepro(text_list, keep_num = True):
    text_list = [text_list ] if type(text_list) == str else text_list 
    
    re_ligne = re.compile(r"\\n")
    re_apo = re.compile(r"^'")
    re_pts = re.compile(r"\.(?![ |\d+])")#\w matches number here with unknown reason
    re_sym = re.compile(r'=')
    re_num = re.compile(r" \d+([,\. /]\d+)?")
    
    
    res = []
    for text in text_list:
        clean_n = re.sub(re_ligne," \n", text)
        clean_apo = re.sub(re_apo,'',clean_n)
        clean_pts = re.sub(re_pts,'. ',clean_apo)
        clean_sym = re.sub(re_sym, ' = ', clean_pts)

        res.append(clean_sym) if(keep_num) else res.append(re.sub(re_num, ' NUM ', clean_sym))
    return res[0] if len(res) == 1 else res


#if __name__ == '__main__':
#    test_num = """Il s’agit d’une patiente âgée de 45 ans, à j3 d’un accouchement, Apgar 8/10 sur une grossesse estimée à 37 semaines abdominales diffuses évoluant depuis trois jours, une fièvre à 38,5 °C, une tension artérielle à 90/60 mm Hg et un pouls à 88 battement/min.une hyperleucocytose à 17 610 éléments/mm3. une aspiration de 1000 ml d’un liquide,tricuspide à 35mm et PAPS à 147mmHg, d'environ 10 à 20mm, (Hb à 8g/dl, TCMH à 25 pg/ml."""
#    print(prepro(test_num))
    
