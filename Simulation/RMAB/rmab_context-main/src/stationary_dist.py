import numpy as np

#note: the matrix is row stochastic.
#A markov chain transition will correspond to left multiplying by a row vector.
'''Q = np.array([
    [0.547945205, 0.374429224, 0.077625571], # G state action 1
    [0.450867052, 0.419075145, 0.130057803], # M state action 1
    [0.266903915, 0.483985765, 0.24911032] # B state action 1
    ])

Q = np.array([
    [0.372651357, 0.424843424, 0.202505219], # G state action 0
    [0.280178838, 0.439642325, 0.280178838], # M state action 0
    [0.132827324, 0.330170778, 0.537001898] # B state action 0
    ])

Q = np.array([
                [0.786078098, 0.198641766, 0.015280136],  # G state action 1
                [0.680147059, 0.275735294, 0.044117647],  # M state action 1
                [0.659574468, 0.29787234, 0.042553191]  # B state action 1
    ])

Q = np.array([
                [0.676666667, 0.283333333, 0.04], # G state action 0
                [0.550682261, 0.34502924, 0.104288499], # M state action 0
                [0.317073171, 0.422764228, 0.260162602], # B state action 0
    ])

Q = np.array([
                [0.715859031, 0.248898678, 0.035242291],  # G state action 1
                [0.646634615, 0.283653846, 0.069711538],  # M state action 1
                [0.464285714, 0.378571429, 0.157142857]  # B state action 1
    ])
    
Q = np.array([
                [0.525381962, 0.364218827, 0.110399211], # G state action 0
                [0.339236303, 0.427780852, 0.232982844], # M state action 0
                [0.15529623, 0.337522442, 0.507181329], # B state action 0
    ])

Q = np.array([
                [0.729323308, 0.242105263, 0.028571429],  # G state action 1
                [0.659090909, 0.292613636, 0.048295455],  # M state action 1
                [0.544303797, 0.35443038, 0.101265823]  # B state action 1
    ])

Q = np.array([
                [0.595296026, 0.343876723, 0.060827251], # G state action 0
                [0.534405145, 0.372347267, 0.093247588], # M state action 0
                [0.429042904, 0.488448845, 0.082508251], # B state action 0
    ])

Q = np.array([
                        [0.66576087, 0.298913043, 0.035326087],  # G state action 1
                        [0.625, 0.311111111, 0.063888889],  # M state action 1
                        [0.495575221, 0.398230088, 0.10619469]  # B state action 1
    ])

Q = np.array([
                        [0.5203357, 0.378954164, 0.100710136], # G state action 0
                        [0.385703064, 0.422623723, 0.191673213], # M state action 0
                        [0.22160149, 0.43575419, 0.34264432], # B state action 0
    ])

Q = np.array([
                        [0.849462366, 0.144393241, 0.006144393], # G state action 1
                        [0.638924456, 0.298335467, 0.062740077], # M state action 0
                        [0.481481481, 0.361111111, 0.157407407], # B state action 0
    ])

Q = np.array([
                        [0.755652519, 0.217770726, 0.026576755], # G state action 0
                        [0.638924456, 0.298335467, 0.062740077], # M state action 0
                        [0.481481481, 0.361111111, 0.157407407], # B state action 0
    ])

Q = np.array([
                        [0.798387097, 0.186827957, 0.014784946],  # G state action 1
                        [0.730245232, 0.226158038, 0.04359673],  # M state action 1
                        [0.641304348, 0.315217391, 0.043478261]  # B state action 1
    ])

Q = np.array([
                        [0.648440121, 0.290841999, 0.06071788], # G state action 0
                        [0.507014028, 0.372745491, 0.120240481], # M state action 0
                        [0.357995227, 0.398568019, 0.243436754], # B state action 0
    ])

Q = np.array([
                        [0.769123783, 0.219749652, 0.011126565],  # G state action 1
                        [0.739413681, 0.2247557, 0.035830619],  # M state action 1
                        [0.6, 0.383333333, 0.016666667]  # B state action 1
    ])

Q = np.array([
                        [0.672443674, 0.284228769, 0.043327556], # G state action 0
                        [0.593724859, 0.333065165, 0.073209976], # M state action 0
                        [0.45631068, 0.36407767, 0.17961165], # B state action 0
    ])

Q = np.array([
                        [0.829824561, 0.159649123, 0.010526316],  # G state action 1
                        [0.717948718, 0.256410256, 0.025641026],  # M state action 1
                        [0.8, 0.166666667, 0.033333333]  # B state action 1
    ])

Q = np.array([
                        [0.705641492, 0.262056415, 0.032302093], # G state action 0
                        [0.609489051, 0.323600973, 0.066909976], # M state action 0
                        [0.463235294, 0.382352941, 0.154411765], # B state action 0
    ])

Q = np.array([
                        [0.882440476, 0.114583333, 0.00297619],  # G state action 1
                        [0.878504673, 0.112149533, 0.009345794],  # M state action 1
                        [0.727272727, 0.272727273, 0]  # B state action 1
    ])

Q = np.array([
                        [0.82731671, 0.15630815, 0.01637514], # G state action 0
                        [0.760589319, 0.222836096, 0.016574586], # M state action 0
                        [0.543478261, 0.391304348, 0.065217391], # B state action 0
    ])

Q = np.array([
                        [0.862767154, 0.132733408, 0.004499438],  # G state action 1
                        [0.76754386, 0.206140351, 0.026315789],  # M state action 1
                        [0.708333333, 0.25, 0.041666667]  # B state action 1
    ])

Q = np.array([
                        [0.789242591, 0.192371021, 0.018386389], # G state action 0
                        [0.71192053, 0.247240618, 0.040838852], # M state action 0
                        [0.615384615, 0.288461538, 0.096153846], # B state action 0
    ])

Q = np.array([
                        [0.840101523, 0.154822335, 0.005076142],  # G state action 1
                        [0.800884956, 0.181415929, 0.017699115],  # M state action 1
                        [0.666666667, 0.285714286, 0.047619048]  # B state action 1
    ])'''

Q = np.array([
                        [0.7658831, 0.215057179, 0.01905972], # G state action 0
                        [0.694063927, 0.276255708, 0.029680365], # M state action 0
                        [0.611111111, 0.277777778, 0.111111111], # B state action 0
    ])
#We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
evals, evecs = np.linalg.eig(Q.T)
evec1 = evecs[:,np.isclose(evals, 1)]

#Since np.isclose will return an array, we've indexed with an array
#so we still have our 2nd axis.  Get rid of it, since it's only size 1.
evec1 = evec1[:,0]

stationary = evec1 / evec1.sum()

#eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
stationary = stationary.real

#print(len(stationary))
print(stationary)
