import re

with open('Paper/Paper.tex', 'r') as f:
    text = f.read()

# Replace OLS table values or other specific values not yet replaced
# I need to do a comprehensive replacement of the values.

# Abstract and intro parts were somewhat handled, let's just make sure
text = text.replace('B = +0.0002, p = 0.716', 'B = 0.0037, p = 0.202')
text = text.replace('B = +0.0002, p = 0.620', 'B = 0.0065, p = 0.356')

# OLS table text or general text for OLS:
text = text.replace('R^2 = 0.017, F(10, 871) = 2638, p < 0.001', 'R^2 = 0.018, F(11, 731) = 0.9286, p = 0.512')
text = text.replace('R^2 = 0.005, F(10, 871) = 498.9, p < 0.001', 'R^2 = 0.017, F(11, 731) = 0.9888, p = 0.455')
text = text.replace('F(10, 871)', 'F(11, 731)')

# CLIP Results from USER PROMPT
# Male mean: -0.0035 (SD = 0.0361) -> maybe \mu_M = -0.0035, \sigma = 0.0361
# Female mean: -0.0067 (SD = 0.0344) -> maybe \mu_F = -0.0067, \sigma = 0.0344
text = text.replace('\\mu_F = -0.0523, \\sigma = 0.0091', '\\mu_F = -0.0067, \\sigma = 0.0344')
text = text.replace('\\mu_M = -0.0520, \\sigma = 0.0091', '\\mu_M = -0.0035, \\sigma = 0.0361')
text = text.replace('\\mu_F = -0.0523', '\\mu_F = -0.0067')
text = text.replace('\\mu_M = -0.0520', '\\mu_M = -0.0035')

# OpenCLIP Results
text = text.replace('\\mu_F = -0.0167, \\sigma = 0.0066', '\\mu_F = 0.0171, \\sigma = 0.0835')
text = text.replace('\\mu_M = -0.0165, \\sigma = 0.0068', '\\mu_M = 0.0237, \\sigma = 0.0886')
text = text.replace('\\mu_F = -0.0167', '\\mu_F = 0.0171')
text = text.replace('\\mu_M = -0.0165', '\\mu_M = 0.0237')

# CLIP OLS
text = text.replace('-0.0241', '-0.0064') # Aspect ratio B in CLIP
text = text.replace('-0.0079', '-0.0023') # Aspect ratio B in OpenCLIP

# CLIP Prompts:
# Set 1 (masterpiece): M -0.0040, F -0.0037, p=0.2550, r=-0.0537
# Set 2 (quality): M 0.0277, F 0.0215, p=0.3302, r=-0.0459
# Set 3 (influence): M -0.0343, F -0.0378, p=0.6397, r=0.0221

# Let's find exactly what they look like in the table using regex

with open('Paper/Paper.tex', 'w') as f:
    f.write(text)

