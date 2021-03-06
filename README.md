# ECON 815: Computational Methods for Economists (Fall 2017) #

|  | [Jason DeBacker](http://jasondebacker.com) |
|--------------|--------------------------------------------------------------|
| Email | [jason.debacker@moore.sc.edu](mailto:jason.debacker@moore.sc.edu) |
| Office | 427B DMSB |
| Office Hours | T 2:45-4:45pm, Th 9:00-11:00am |
| GitHub | [jdebacker](https://github.com/jdebacker) |

* **Meeting day/time**: T,Th 1:15-2:30pm, DMSB, Room 121
* Office hours also available by appointment


## Course description ##

This course is designed to introduce PhD students to software applications and computational techniques to make them productive researchers. Students will be exposed to leading open source software packages (Python, R) and techniques for numerical computing and data analysis. The course will be taught through the application of these software packages and methods to economic research in applied microeconomics and quantitative macroeconomics.


## Course Objectives and Learning Outcomes ##

In this course students, through lecture and application, students will learn about:
* Software to increase research productivity including:
	* TeX
	* git
	* Python
	* R
* How to write custom estimation routines and use packages written by others for:
	* Geneneralized method of moments estimators
	* Maximum likelihood estimators
	* Maximum score estimators
	* Reduced form estimators such as regression discontinuity design
	* Simulated method of moment estimators
* Computational methods to:
	* Optimize and find roots of functions
	* Solve dynamic programming problems (disceret and continous choice)
	* Solve general equilibrium, heterogeneous agent models
	* Perform Monte Carlo simulations
	* Bootstrap standard errors
	* Compute numerical derivatives
	* Use just-in-time compilation for efficient computation
	* Run computations in parallel using multiple processors
* Methods to gather and handle data including:
	* Costs and benefits of different data structures
	* Using APIs
	* Web scraping
* Coding and collaboration techniques such as:
	* Writing modular code with functions and objects
	* Collaboration tools for writing code using [Git](https://git-scm.com/) and [GitHub.com](https://github.com/).
	* Best practices for Python coding ([PEP 8](https://www.python.org/dev/peps/pep-0008/))


## Grades ##

Grades will be based on the categories listed below with the corresponding weights.

Assignment                   | Points |   Percent  |
-----------------------------|--------|------------|
Problem Sets                 |   90   |    90%   |
Class Participation                |   10   |    10.0%   |
**Total points**             | **100** | **100.0%** |

* **Homework:** I will assign 9 problem sets throughout the semester.
	* You must write and submit your own computer code, although I encourage you to collaborate with your fellow students. I **DO NOT** want to see a bunch of copies of identical code. I **DO** want to see each of you learning how to code these problems so that you could do it on your own.
	* Problem set solutions, both written and code portions, will be turned in via a pull request from your private [GitHub.com](https://git-scm.com/) repository which is a fork of the class master repository on my account. (You will need to set up a GitHub account if you do not already have one.)
	* Written solutions must be submitted as pdf documents or Jupyter Notebooks.
	* Problem sets will be due on the day listed in the Daily Course Schedule section of this syllabus (see below) unless otherwise specified. Late homework will not be graded.



## Daily Course Schedule ##

| Date     | Day | Topic                                  | Due    |
|----------|-----|----------------------------------------|--------|
| Aug. 24  | Th  | [Work Flow, Productivity Software](https://github.com/jdebacker/CompEcon_Fall17/tree/master/Productivity)       |        |
| Aug. 29  | T   | [Work Flow, Productivity Software](https://github.com/jdebacker/CompEcon_Fall17/tree/master/Productivity)         |        |
| Aug. 31  | Th  | [Python/Object Oriented Programming](https://github.com/jdebacker/CompEcon_Fall17/tree/master/Python)                             | [PS #1](https://github.com/jdebacker/CompEcon_Fall17/blob/master/Productivity/PS1.pdf)  |
| Sept. 5  | T   | [Python/Object Oriented Programming](https://github.com/jdebacker/CompEcon_Fall17/tree/master/Python)                             |        |
| Sept. 7  | Th  | [Functions, Optimizers, Root-Finders](https://github.com/jdebacker/CompEcon_Fall17/tree/master/Functions)    |        |
| Sept. 12 | T   | [Functions, Optimizers, Root-Finders](https://github.com/jdebacker/CompEcon_Fall17/tree/master/Functions)     | [PS #2](https://github.com/jdebacker/CompEcon_Fall17/blob/master/Python/PS2.pdf)  |
| Sept. 14 | Th  | [Functions, Optimizers, Root-Finders](https://github.com/jdebacker/CompEcon_Fall17/tree/master/Functions)      |        |
| Sept. 19 | T   | [Functions, Optimizers, Root-Finders](https://github.com/jdebacker/CompEcon_Fall17/tree/master/Functions)      |        |
| Sept. 21 | Th  | [Conditionals/Loops](https://github.com/jdebacker/CompEcon_Fall17/tree/master/LoopConditional)                     |        |
| Sept. 26 | T   | [Conditionals/Loops](https://github.com/jdebacker/CompEcon_Fall17/tree/master/LoopConditional)                     | [PS #3](https://github.com/jdebacker/CompEcon_Fall17/blob/master/Functions/PS3.pdf)  |
| Sept. 28 | Th   | [Conditionals/Loops](https://github.com/jdebacker/CompEcon_Fall17/tree/master/LoopConditional)                     | |
| Oct. 3   | T   | [Visualization](https://github.com/jdebacker/CompEcon_Fall17/tree/master/Visualization)                          |   |
| Oct. 5   | Th   | [Visualization](https://github.com/jdebacker/CompEcon_Fall17/tree/master/Visualization)                          |   |
| Oct. 10   | T  | [Econometrics in Python and R](https://github.com/jdebacker/CompEcon_Fall17/tree/master/R)           |        |
| Oct. 12  | Th   | [Econometrics in Python and R](https://github.com/jdebacker/CompEcon_Fall17/tree/master/R)           | [PS #4](https://github.com/jdebacker/CompEcon_Fall17/blob/master/LoopConditional/PS4.pdf)  |
| Oct. 17  | T  | [Econometrics in Python and R](https://github.com/jdebacker/CompEcon_Fall17/tree/master/R)           |        |
| Oct. 19  | Th  | No class, Fall Break                   |        |
| Oct. 24  | T   | [Web scraping/APIs to gather data](https://github.com/jdebacker/CompEcon_Fall17/tree/master/WebData)       | [PS #5](https://github.com/jdebacker/CompEcon_Fall17/blob/master/Visualization/PS5.pdf)       |
| Oct. 26  | Th  | [Web scraping/APIs to gather data](https://github.com/jdebacker/CompEcon_Fall17/tree/master/WebData)       |   |
| Oct. 31  | T   | [Web scraping/APIs to gather data](https://github.com/jdebacker/CompEcon_Fall17/tree/master/WebData)       | [PS #6](https://github.com/jdebacker/CompEcon_Fall17/blob/master/R/PS6.pdf)     |
| Nov. 2   | Th  | [Dynamic Programming](https://github.com/jdebacker/CompEcon_Fall17/tree/master/DynamicProgramming)       |        |
| Nov. 7   | T   | [Dynamic Programming](https://github.com/jdebacker/CompEcon_Fall17/tree/master/DynamicProgramming)  |        |
| Nov. 9   | Th  | TBD                                    |        |
| Nov. 14  | T   | [Markov Chains](https://github.com/jdebacker/CompEcon_Fall17/tree/master/DynamicProgramming)  | [PS #7](https://github.com/jdebacker/CompEcon_Fall17/blob/master/WebData/PS7.pdf)  |
| Nov. 16  | Th  | [Stationary Distributions](https://github.com/jdebacker/CompEcon_Fall17/tree/master/DynamicProgramming)  |        |
| Nov. 21  | T   | [Solving GE Models](https://github.com/jdebacker/CompEcon_Fall17/tree/master/DynamicProgramming)  |        |
| Nov. 23  | Th  | No class, Thanksgiving                 |        |
| Nov. 28  | T   | [Simulation Methods](https://github.com/jdebacker/CompEcon_Fall17/blob/master/SMM/)                     |        |
| Nov. 30  | Th  | [Simulation Methods](https://github.com/jdebacker/CompEcon_Fall17/blob/master/SMM/)                     | |
| Dec. 4   | M  | No class meeting                     | [PS # 8](https://github.com/jdebacker/CompEcon_Fall17/blob/master/DynamicProgramming/PS8.pdf) |
| Dec. 5   | T   | [Discerete Choice Dynamic Programming](https://github.com/jdebacker/CompEcon_Fall17/tree/master/DynamicProgramming)                     |        |
| Dec. 7   | Th  | [Parallel Processing](https://github.com/jdebacker/CompEcon_Fall17/blob/master/Multiprocessing)                    |        |
|          |     |                                        |        |
| Dec. 14  | Th  | No Final Exam - project due            | [PS #9](https://github.com/jdebacker/CompEcon_Fall17/blob/master/SMM/PS9.pdf)  |


## Helpful Links ##

* [QuantEcon](https://quantecon.org)
* [Notes on Machine Learning & Artificial Intelligence](https://chrisalbon.com) by Chris Albon


## Reasonable Accommodations for Students with Disabilities: ##

If you have any condition, such as a physical or learning disability, which will make it difficult for you to carry out the work as I have outlined it or which will require academic accommodations, please notify me through email AND in person with the appropriate documentation within the first two weeks of the course. Please also copy the course TA to this message.
