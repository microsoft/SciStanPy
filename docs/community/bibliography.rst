Bibliography and References
===========================

This bibliography provides key references for Bayesian statistics, probabilistic programming, and scientific computing that inform and support SciStanPy's design and capabilities.

Core Bayesian Statistics
------------------------

**Foundational Texts**

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).
*Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.

    Comprehensive treatment of Bayesian methods with practical applications.
    The gold standard reference for Bayesian data analysis.

McElreath, R. (2020).
*Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). Chapman and Hall/CRC.

    Accessible introduction to Bayesian thinking with emphasis on understanding over mechanical application.

Kruschke, J. K. (2014).
*Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan* (2nd ed.). Academic Press.

    Practical guide to Bayesian analysis with focus on implementation and interpretation.

**Theoretical Foundations**

Bernardo, J. M., & Smith, A. F. M. (2000).
*Bayesian Theory*. John Wiley & Sons.

    Rigorous mathematical treatment of Bayesian theory and decision theory.

Robert, C. P. (2007).
*The Bayesian Choice: From Decision-Theoretic Foundations to Computational Implementation* (2nd ed.). Springer.

    Comprehensive coverage of Bayesian decision theory and computational methods.

Probabilistic Programming
-------------------------

**Stan and MCMC**

Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., ... & Riddell, A. (2017).
Stan: A probabilistic programming language.
*Journal of Statistical Software*, 76(1), 1-32.

    Official description of the Stan probabilistic programming language and its capabilities.

Betancourt, M. (2017).
A conceptual introduction to Hamiltonian Monte Carlo.
arXiv preprint arXiv:1701.02434.

    Excellent tutorial on the Hamiltonian Monte Carlo method used by Stan.

Hoffman, M. D., & Gelman, A. (2014).
The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo.
*Journal of Machine Learning Research*, 15(1), 1593-1623.

    Description of the NUTS algorithm used in Stan for adaptive MCMC sampling.

**Variational Inference**

Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017).
Variational inference: A review for statisticians.
*Journal of the American Statistical Association*, 112(518), 859-877.

    Comprehensive review of variational inference methods for approximate Bayesian computation.

Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017).
Automatic differentiation variational inference.
*Journal of Machine Learning Research*, 18(1), 430-474.

    Foundation for automatic variational inference methods.

Scientific Computing and Software
---------------------------------

**Scientific Python Ecosystem**

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020).
Array programming with NumPy.
*Nature*, 585(7825), 357-362.

    Description of NumPy, the fundamental package for scientific computing in Python.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & SciPy 1.0 Contributors. (2020).
SciPy 1.0: fundamental algorithms for scientific computing in Python.
*Nature Methods*, 17(3), 261-272.

    Overview of SciPy library for scientific computing algorithms.

**PyTorch and Automatic Differentiation**

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).
PyTorch: An imperative style, high-performance deep learning library.
*Advances in Neural Information Processing Systems*, 32, 8024-8035.

    Description of PyTorch framework used for automatic differentiation in SciStanPy.

Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2017).
Automatic differentiation in machine learning: a survey.
*Journal of Machine Learning Research*, 18(1), 5595-5637.

    Comprehensive survey of automatic differentiation techniques.

Model Selection and Validation
------------------------------

**Information Criteria and Cross-Validation**

Vehtari, A., Gelman, A., & Gabry, J. (2017).
Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
*Statistics and Computing*, 27(5), 1413-1432.

    Practical guide to model comparison using LOO-CV and WAIC.

Watanabe, S. (2010).
Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory.
*Journal of Machine Learning Research*, 11, 3571-3594.

    Theoretical foundation for WAIC as an information criterion.

Gelman, A., Hwang, J., & Vehtari, A. (2014).
Understanding predictive information criteria for Bayesian models.
*Statistics and Computing*, 24(6), 997-1016.

    Conceptual explanation of information criteria for model selection.

**Model Checking**

Gelman, A., Meng, X. L., & Stern, H. (1996).
Posterior predictive assessment of model fitness via realized discrepancies.
*Statistica Sinica*, 6(4), 733-760.

    Foundation for posterior predictive checking methodology.

Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019).
Visualization in Bayesian workflow.
*Journal of the Royal Statistical Society: Series A*, 182(2), 389-402.

    Best practices for visualizing Bayesian analysis results.

Scientific Applications
----------------------

**Parameter Estimation**

Sivia, D., & Skilling, J. (2006).
*Data Analysis: A Bayesian Tutorial* (2nd ed.). Oxford University Press.

    Practical guide to Bayesian data analysis with scientific applications.

Gregory, P. C. (2005).
*Bayesian Logical Data Analysis for the Physical Sciences*. Cambridge University Press.

    Application of Bayesian methods to physical sciences with detailed examples.

**Experimental Design**

Chaloner, K., & Verdinelli, I. (1995).
Bayesian experimental design: A review.
*Statistical Science*, 10(3), 273-304.

    Review of Bayesian approaches to experimental design optimization.

Ryan, E. G., Drovandi, C. C., McGree, J. M., & Pettitt, A. N. (2016).
A review of modern computational algorithms for Bayesian optimal design.
*International Statistical Review*, 84(1), 128-154.

    Modern computational approaches to optimal experimental design.

**Hierarchical Modeling**

Gelman, A., & Hill, J. (2006).
*Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.

    Comprehensive treatment of hierarchical modeling with practical examples.

Raudenbush, S. W., & Bryk, A. S. (2002).
*Hierarchical Linear Models: Applications and Data Analysis Methods* (2nd ed.). Sage Publications.

    Applied perspective on hierarchical linear models.

Domain-Specific Applications
---------------------------

**Physics and Astronomy**

Trotta, R. (2008).
Bayes in the sky: Bayesian inference and model selection in cosmology.
*Contemporary Physics*, 49(2), 71-104.

    Application of Bayesian methods to cosmological parameter estimation.

Hogg, D. W., Bovy, J., & Lang, D. (2010).
Data analysis recipes: Fitting a model to data.
arXiv preprint arXiv:1008.4686.

    Practical guide to model fitting in astronomical applications.

**Biology and Medicine**

Spiegelhalter, D. J., Abrams, K. R., & Myles, J. P. (2004).
*Bayesian Approaches to Clinical Trials and Health-Care Evaluation*. John Wiley & Sons.

    Application of Bayesian methods to clinical research and healthcare evaluation.

Beaumont, M. A., Zhang, W., & Balding, D. J. (2002).
Approximate Bayesian computation in population genetics.
*Genetics*, 162(4), 2025-2035.

    Introduction to approximate Bayesian computation methods in genetics.

**Chemistry and Materials Science**

Angelikopoulos, P., Papadimitriou, C., & Koumoutsakos, P. (2012).
Bayesian uncertainty quantification and propagation in molecular dynamics simulations: a high performance computing framework.
*Journal of Chemical Physics*, 137(14), 144103.

    Application of Bayesian methods to molecular dynamics and materials simulation.

**Environmental Science**

Clark, J. S. (2005).
Why environmental scientists are becoming Bayesians.
*Ecology Letters*, 8(1), 2-14.

    Overview of Bayesian applications in environmental and ecological research.

Computational Statistics
------------------------

**MCMC Methods**

Brooks, S., Gelman, A., Jones, G., & Meng, X. L. (Eds.). (2011).
*Handbook of Markov Chain Monte Carlo*. CRC Press.

    Comprehensive reference for MCMC methods and applications.

Robert, C., & Casella, G. (2013).
*Monte Carlo Statistical Methods* (2nd ed.). Springer.

    Theoretical and practical treatment of Monte Carlo methods.

**Diagnostics and Convergence**

Gelman, A., & Rubin, D. B. (1992).
Inference from iterative simulation using multiple sequences.
*Statistical Science*, 7(4), 457-472.

    Introduction of the R-hat statistic for assessing MCMC convergence.

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC.
*Bayesian Analysis*, 16(2), 667-718.

    Modern improvements to convergence diagnostics.

Software and Implementation
--------------------------

**Probabilistic Programming Languages**

van de Meent, J. W., Paige, B., Yang, H., & Wood, F. (2018).
An introduction to probabilistic programming.
arXiv preprint arXiv:1809.10756.

    Survey of probabilistic programming languages and paradigms.

Goodman, N. D., Mansinghka, V. K., Roy, D. M., Bonawitz, K., & Tenenbaum, J. B. (2008).
Church: a language for generative models.
In *Proceedings of the 24th Conference on Uncertainty in Artificial Intelligence* (pp. 220-229).

    Early work on probabilistic programming language design.

**Scientific Software Development**

Wilson, G., Aruliah, D. A., Brown, C. T., Hong, N. P. C., Davis, M., Guy, R. T., ... & Wilson, P. (2014).
Best practices for scientific computing.
*PLoS Biology*, 12(1), e1001745.

    Guidelines for developing reliable scientific software.

Jiménez, R. C., Kuzak, M., Alhamdoosh, M., Barker, M., Batut, B., Borg, M., ... & Crouch, S. (2017).
Four simple recommendations to encourage best practices in research software.
*F1000Research*, 6, 876.

    Practical recommendations for scientific software development.

Related Software and Packages
-----------------------------

**PyMC**

Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016).
Probabilistic programming in Python using PyMC3.
*PeerJ Computer Science*, 2, e55.

    Description of PyMC, another popular Bayesian modeling package in Python.

**TensorFlow Probability**

Dillon, J. V., Langmore, I., Tran, D., Brevdo, E., Vasudevan, S., Moore, D., ... & Saurous, R. A. (2017).
TensorFlow Distributions.
arXiv preprint arXiv:1711.10604.

    Description of TensorFlow Probability library for probabilistic modeling.

**Edward**

Tran, D., Kucukelbir, A., Dieng, A. B., Rudolph, M., Liang, D., & Blei, D. M. (2016).
Edward: A library for probabilistic modeling, inference, and criticism.
arXiv preprint arXiv:1610.09787.

    Description of Edward library for probabilistic programming.

Tutorials and Educational Resources
----------------------------------

**Online Resources**

Betancourt, M. (2018).
*A Conceptual Introduction to Hamiltonian Monte Carlo*.
Available: https://arxiv.org/abs/1701.02434

    Accessible introduction to the theory behind modern MCMC methods.

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2019).
*Bayesian workflow*.
Available: https://arxiv.org/abs/1507.08050

    Practical guide to the complete Bayesian modeling workflow.

**Course Materials**

Statistical Rethinking Course Materials: https://github.com/rmcelreath/stat_rethinking_2023

    Course materials for Richard McElreath's Statistical Rethinking course.

Stan Case Studies: https://mc-stan.org/users/documentation/case-studies

    Collection of detailed case studies using Stan for various applications.

Citing SciStanPy
---------------

If you use SciStanPy in your research, please cite:

    [Author], [Author], et al. (2024).
    SciStanPy: Intuitive Bayesian Modeling for Scientists.
    *Software*.
    Available: https://github.com/microsoft/SciStanPy

For specific versions:

    [Author], [Author], et al. (2024).
    SciStanPy: Intuitive Bayesian Modeling for Scientists (Version 1.0.0).
    *Software*.
    DOI: [DOI if available]

BibTeX entry:

.. code-block:: bibtex

   @software{scistanpy2024,
     title={SciStanPy: Intuitive Bayesian Modeling for Scientists},
     author={[Authors]},
     year={2024},
     version={1.0.0},
     url={https://github.com/microsoft/SciStanPy},
     doi={[DOI if available]}
   }

Contributing to Bibliography
---------------------------

To suggest additions to this bibliography:

1. **Check relevance**: Ensure the reference is directly relevant to Bayesian statistics, scientific computing, or SciStanPy applications
2. **Follow format**: Use consistent citation format with abstracts for key references
3. **Submit via GitHub**: Open an issue or pull request with the suggested addition
4. **Provide context**: Explain why the reference is valuable for SciStanPy users

This bibliography is maintained by the SciStanPy community and updated regularly to include new relevant publications and resources.
