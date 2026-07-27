import re

with open('Paper/Paper.tex', 'r') as f:
    text = f.read()

# Replace general counts
text = text.replace('N = 882', 'N = 743')
text = text.replace('n = 625', 'n = 534')
text = text.replace('n = 257', 'n = 209')
text = text.replace('N=882', 'N=743')
text = text.replace('n=625', 'n=534')
text = text.replace('n=257', 'n=209')
text = text.replace('625 male-attributed, 257 female-attributed', '534 male-attributed, 209 female-attributed')
text = text.replace('Male $n = 625$, Female $n = 257$', 'Male $n = 534$, Female $n = 209$')
text = text.replace('Male $n=625$, Female $n=257$', 'Male $n=534$, Female $n=209$')
text = text.replace('Male ($n=625$)', 'Male ($n=534$)')
text = text.replace('Female ($n=257$)', 'Female ($n=209$)')
text = text.replace('625 Male, 257 Female', '534 Male, 209 Female')

# Section 3.2 Evaluation Cohort Stratification
text = text.replace('70.86\\%', '71.87\\%') # 534/743
text = text.replace('29.14\\%', '28.13\\%') # 209/743
# 534/209 = 2.55x, let's just say >2.5\times
text = text.replace('\\>2.4\\times', '\\>2.5\\times')
text = text.replace('\>2.4\\times', '\>2.5\\times')
text = text.replace('>2.4\\times', '>2.5\\times')

# Table 2 counts
# \textit{Subtotal Named Works} & \textit{882} & \textit{100.00\%} \\
text = text.replace('\\textit{882} & \\textit{100.00\\%}', '\\textit{743} & \\textit{100.00\\%}')


# Abstract replacements
text = text.replace(
    '\\mu_F = -0.0523, \\sigma = 0.0091$ vs $\\mu_M = -0.0520, \\sigma = 0.0091$; $U = 80,904.00$, $p = 0.8642$, rank-biserial $r = -0.0074$',
    '\\mu_F = -0.0067, \\sigma = 0.0344$ vs $\\mu_M = -0.0035, \\sigma = 0.0361$; $U = 59,307.00$, $p = 0.1829$, rank-biserial $r = -0.0628$'
)
text = text.replace(
    '\\mu_F = -0.0167, \\sigma = 0.0066$ vs $\\mu_M = -0.0165, \\sigma = 0.0068$; $U = 82,591.00$, $p = 0.5076$, $r = -0.0284$',
    '\\mu_F = 0.0171, \\sigma = 0.0835$ vs $\\mu_M = 0.0237, \\sigma = 0.0886$; $U = 59,867.00$, $p = 0.1224$, $r = -0.0728$'
)
text = text.replace(
    'R^2 = 0.017$, $F(10, 871) = 2638$, $p < 0.001$ for OpenAI CLIP',
    'R^2 = 0.018$, $F(11, 731) = 0.9286$, $p = 0.512$ for OpenAI CLIP'
)
text = text.replace(
    'R^2 = 0.005$, $F(10, 871) = 498.9$, $p < 0.001$ for OpenCLIP',
    'R^2 = 0.017$, $F(11, 731) = 0.9888$, $p = 0.455$ for OpenCLIP'
)
text = text.replace(
    'B = -0.0241, p < 0.001$ in CLIP; $B = -0.0079, p < 0.001$ in OpenCLIP',
    'B = -0.0064, p = 0.141$ in CLIP; $B = -0.0023, p = 0.516$ in OpenCLIP'
)
text = text.replace(
    'B = +0.0002, p = 0.716$ in CLIP; $B = +0.0002, p = 0.620$ in OpenCLIP',
    'B = 0.0037, p = 0.202$ in CLIP; $B = 0.0065, p = 0.356$ in OpenCLIP'
)

# Intro list
text = text.replace(
    '\\mu_F = -0.0523$ vs $\\mu_M = -0.0520, U = 80,904.00, p = 0.8642, r = -0.0074$',
    '\\mu_F = -0.0067$ vs $\\mu_M = -0.0035, U = 59,307.00, p = 0.1829, r = -0.0628$'
)
text = text.replace(
    '\\mu_F = -0.0167$ vs $\\mu_M = -0.0165, U = 82,591.00, p = 0.5076, r = -0.0284$',
    '\\mu_F = 0.0171$ vs $\\mu_M = 0.0237, U = 59,867.00, p = 0.1224, r = -0.0728$'
)
text = text.replace(
    'R^2 = 0.017, F(10, 871) = 2638, p < 0.001$ for CLIP',
    'R^2 = 0.018, F(11, 731) = 0.9286, p = 0.512$ for CLIP'
)
text = text.replace(
    'R^2 = 0.005, F(10, 871) = 498.9, p < 0.001$ for OpenCLIP',
    'R^2 = 0.017, F(11, 731) = 0.9888, p = 0.455$ for OpenCLIP'
)
text = text.replace(
    'B = -0.0241, p < 0.001$',
    'B = -0.0064, p = 0.141$'
)

# Abstract and conclusion
text = text.replace(
    'U = 80,904.00, p = 0.8642, r = -0.0074',
    'U = 59,307.00, p = 0.1829, r = -0.0628'
)
text = text.replace(
    'U = 82,591.00, p = 0.5076, r = -0.0284',
    'U = 59,867.00, p = 0.1224, r = -0.0728'
)

# Text and Tables 2, 5 etc

with open('Paper/Paper.tex', 'w') as f:
    f.write(text)

