# Results of Statistical Analysis
analysis_results = '''
Descriptive Statistics for Stability and Loudness:
       MDVP:Fo(Hz)  MDVP:Fhi(Hz)  MDVP:Flo(Hz)
count   195.000000    195.000000    195.000000
mean      0.383623      0.193841      0.292748
std       0.240959      0.186761      0.250564
min       0.000000      0.000000      0.000000
25%       0.170220      0.066786      0.108323
50%       0.351961      0.150411      0.223606
75%       0.549775      0.249162      0.429160
max       1.000000      1.000000      1.000000

Normality check for MDVP:Fo(Hz):
Shapiro-Wilk Test for MDVP:Fo(Hz): Stat=0.9370526258264308, p-value=1.733783802845246e-07
MDVP:Fo(Hz) does not follow a normal distribution.

Normality check for MDVP:Fhi(Hz):
Shapiro-Wilk Test for MDVP:Fhi(Hz): Stat=0.7228382441400187, p-value=9.296039686721633e-18
MDVP:Fhi(Hz) does not follow a normal distribution.

Normality check for MDVP:Flo(Hz):
Shapiro-Wilk Test for MDVP:Flo(Hz): Stat=0.8608445971669065, p-value=2.274371506176558e-12
MDVP:Flo(Hz) does not follow a normal distribution.

Comparing groups for MDVP:Fo(Hz):
T-Test for MDVP:Fo(Hz): Stat=5.769452074779705, p-value=3.121919402836263e-08
There is a significant difference in MDVP:Fo(Hz) between healthy individuals and Parkinson's patients.

Comparing groups for MDVP:Fhi(Hz):
T-Test for MDVP:Fhi(Hz): Stat=2.340567250831287, p-value=0.020275669142798634
There is a significant difference in MDVP:Fhi(Hz) between healthy individuals and Parkinson's patients.

Comparing groups for MDVP:Flo(Hz):
T-Test for MDVP:Flo(Hz): Stat=5.710768359712574, p-value=4.1970041821535536e-08
There is a significant difference in MDVP:Flo(Hz) between healthy individuals and Parkinson's patients.

Correlation Matrix for Stability and Loudness Columns:
              MDVP:Fo(Hz)  MDVP:Fhi(Hz)  MDVP:Flo(Hz)
MDVP:Fo(Hz)      1.000000      0.400985      0.596546
MDVP:Fhi(Hz)     0.400985      1.000000      0.084951
MDVP:Flo(Hz)     0.596546      0.084951      1.000000
'''