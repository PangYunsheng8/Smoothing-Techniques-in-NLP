# Smoothing techniques commonly used in NLP  
In this notebook, I will introduce several smoothing techniques commonly used in NLP or machine learning algorithms. They are:
- Laplacian (add-one) Smoothing
- Lidstone (add-k) Smoothing
- Absolute Discounting
- Katz Backoff
- Kneser-Ney Smoothing
- Interpolation

## Laplacian (add-one) Smoothing
Laplacian (add-one) smoothing:   
$$ P_{add-1}(w_{i}|w_{i-1}) = \frac{C(w_{i-1}, w_{i}) + 1}{C(w_{i-1}) + |V|}$$  
**Core idea**: Pretend that we have seen each vocab at least once.  
vocabs | counts | unsmoothed probability | laplacian (add-one) smoothing
:-: | :-: | :-: | :-: 
impropriety | 8 | 0.4 | (8+1)/(20+7)= 0.333
offense | 5 | 0.25 | (5+1)/(20+7)= 0.222
damage | 4 | 0.2 | (4+1)/(20+7)= 0.186
deficiencies | 2 | 0.1 | (2+1)/(20+7)= 0.111
outbreak | 1 | 0.05 | (1+1)/(20+7)= 0.074
infirmity | 0 | 0 | (0+1)/(20+7)= 0.037
cephalopods | 0 | 0 | (0+1)/(20+7)= 0.037
**total** | **20** | **1.0** | **1.0**  

## Lidstone (add-k) Smoothing  
Lidstone (add-k) smoothing:   
$$ P_{add-k}(w_{i}|w_{i-1}) = \frac{C(w_{i-1}, w_{i}) + k}{C(w_{i-1}) + k|V|}$$  
**Core idea**: Sometimes adding one is too much, instead, we add k (usually k < 1).  
vocabs | counts | unsmoothed probability | lidstone (add-k) smoothing (k=0.5)
:-: | :-: | :-: | :-: 
impropriety | 8 | 0.4 | (8+0.5)/(20+7*0.5)= 0.363
offense | 5 | 0.25 | (5+0.5)/(20+7*0.5)= 0.234
damage | 4 | 0.2 | (4+0.5)/(20+7*0.5)= 0.191
deficiencies | 2 | 0.1 | (2+0.5)/(20+7*0.5)= 0.106
outbreak | 1 | 0.05 | (1+0.5)/(20+7*0.5)= 0.064
infirmity | 0 | 0 | (0+0.5)/(20+7*0.5)= 0.021
cephalopods | 0 | 0 | (0+0.5)/(20+7*0.5)= 0.021
**total** | **20** | **1.0** | **1.0**  

## Absolute Discounting  
Absolute discounting:  
$$ P_{absolute-discounting}(w_{i}|w_{i-1})=\left\{
\begin{aligned}
\frac{C(w_{i-1}, w_{i}) - D}{C(w_{i-1})}, if \quad C(w_{i-1}, w_{i}) > 0 \\
\alpha(w_{i-1}) / \sum\nolimits_{w_{j}:C(w_{i-1}, w_{j})=0}, otherwise
\end{aligned}
\right.
$$  
**Core idea**: 'Borrows' a fixed probability mass from observed n-gram counts and redistributes it to unseen n-grams.  
$\alpha(w_{i-1})$ is the amount of probability mass that has been discounted for context $w_{i-1}$, in this example, its valuse is (0.1*5)/20.  
$\sum\nolimits_{w_{j}:C(w_{i-1}, w_{j})=0}$ is the count of $C(w_{i-1}, w_{j})=0$, here it is 2.  
vocabs | counts | unsmoothed probability | absolute discounting (d=0.1) | effective counts
:-: | :-: | :-: | :-: | :-: 
impropriety | 8 | 0.4 | (8-0.1)/20=0.395 | 7.9
offense | 5 | 0.25 | (5-0.1)/20=0.245 | 4.9
damage | 4 | 0.2 | (4-0.1)/20=0.195 | 3.9
deficiencies | 2 | 0.1 | (2-0.1)/20=0.095 | 1.9
outbreak | 1 | 0.05 | (1-0.1)/20=0.045 | 0.9
infirmity | 0 | 0 | (0+0.5)/20/2=0.0125 | 0.25
cephalopods | 0 | 0 | (0+0.5)/20/2=0.0125 | 0.25
**total** | **20** | **1.0** | **1.0** | **20**    

## Katz Backoff  
Katz Backoff:  
$$ P_{backoff}(w_{i}|w_{i-1})=\left\{
\begin{aligned}
\frac{C(w_{i-1}, w_{i}) - D}{C(w_{i-1})}, if \quad C(w_{i-1}, w_{i}) > 0 \\
\alpha(w_{i-1}) \times \frac{P(w_{j})}{\sum\nolimits_{w_{j}:C(w_{i-1}, w_{j})=0}{P(w_{j})}}, otherwise
\end{aligned}
\right.
$$  
**Core idea**: Absolute discounting redistributes the probability mass **equally** for all unseen n-grams while Backoff redistributes the mass based on a lower order model (e.g. unigram).  
$\alpha(w_{i-1})$ is also the amount of probability mass that has been discounted for context $w_{i-1}$, in this example, its valuse is (0.1*5)/20.  
$P(w_{i})$ is the unigram probability for $w_{i}$. Suppose here $P(infirmity) = 0.002$, $P(cephalopods) = 0.008$.  
vocabs | counts | unsmoothed probability | backoff | effective counts
:-: | :-: | :-: | :-: | :-: 
impropriety | 8 | 0.4 | (8-0.1)/20=0.395 | 7.9
offense | 5 | 0.25 | (5-0.1)/20=0.245 | 4.9
damage | 4 | 0.2 | (4-0.1)/20=0.195 | 3.9
deficiencies | 2 | 0.1 | (2-0.1)/20=0.095 | 1.9
outbreak | 1 | 0.05 | (1-0.1)/20=0.045 | 0.9
infirmity | 0 | 0 | (0+0.5)/20*0.002/(0.002+0.008)=0.0005 | 0.1
cephalopods | 0 | 0 | (0+0.5)/20*0.008/(0.002+0.008)=0.02 | 0.4
**total** | **20** | **1.0** | **1.0** | **20**  

## Kneser-Ney Smoothing  
Kneser-Ney Smoothing:  
$$ P_{kneser-ney-smoothing}(w_{i}|w_{i-1})=\left\{
\begin{aligned}
\frac{C(w_{i-1}, w_{i}) - D}{C(w_{i-1})}, if \quad C(w_{i-1}, w_{i}) > 0 \\
\alpha(w_{i-1})P_{cont}(w_{i}), otherwise
\end{aligned}
\right.\\
where \quad
P_{cont}(w_{i}) = \frac{|\{w_{i-1}:C(w_{i-1}, w_{i}) > 0\}|}{{\sum_{w_{i}}{|\{w_{i-1}:C(w_{i-1}, w_{i}) > 0\}|}}}
$$  
**Core idea**: Redistribute probability mass based on how many number of different contexts word w has appeared in.  
$\alpha(w_{i-1})$ is also the amount of probability mass that has been discounted for context $w_{i-1}$, in this example, its valuse is (0.1*5)/20.  
Suppose we have the following phrases in the corpus: {A infirmity, B infirmity, C infirmity, D infirmity, A cephalopods}, then  
$|\{w_{i-1}:C(w_{i-1}, w_{i}) > 0\}|$ for $w_{i}$ = infirmity is 4, $P_{cont}(w_{i}=infirmity)$ = 4/(4+1)= 0.8.  
$|\{w_{i-1}:C(w_{i-1}, w_{i}) > 0\}|$ for $w_{i}$ = cephalopods is 1, $P_{cont}(w_{i}=cephalopods)$ = 1/(4+1)= 0.2  
vocabs | counts | unsmoothed probability | kneser-ney smoothing | effective counts
:-: | :-: | :-: | :-: | :-: 
impropriety | 8 | 0.4 | (8-0.1)/20=0.395 | 7.9
offense | 5 | 0.25 | (5-0.1)/20=0.245 | 4.9
damage | 4 | 0.2 | (4-0.1)/20=0.195 | 3.9
deficiencies | 2 | 0.1 | (2-0.1)/20=0.095 | 1.9
outbreak | 1 | 0.05 | (1-0.1)/20=0.045 | 0.9
infirmity | 0 | 0 | (0+0.5)/20*4/(4+1)=0.02 | 0.4
cephalopods | 0 | 0 | (0+0.5)/20*1/(4+1)=0.005 | 0.1
**total** | **20** | **1.0** | **1.0** | **20**  

## Interpolation  
Interpolation:  
$$ 
\begin{aligned}
P_{interpolation}(w_{i}|w_{i-1}, w_{i-2})&=\lambda_{3}P_{3}(w_{i}|w_{i-1}, w_{i-2}) \\
&+\lambda_{2}P_{2}(w_{i}|w_{i-1})\\
&+\lambda_{1}P_{1}(w_{i})\\
where \quad
\sum_{i}{\lambda_{i}} = 1
\end{aligned}
$$  
**Core idea**: Combine different order n-gram models.
