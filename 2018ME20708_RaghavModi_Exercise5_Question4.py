import numpy
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot
import matplotlib.colors

N_j = 250
Y_j = numpy.array([28,53,93,126,172,197])
P_j = Y_j/N_j
X_j = numpy.array([1,2,3,4,5,6])

y = numpy.array([1]*6)
p_set = numpy.where(P_j<0.5)
y[p_set] = 0
x = numpy.array([[1],[2],[3],[4],[5],[6]])

reg_model = LogisticRegression(random_state = 0)
reg_model.fit(x,y)
B_0 = reg_model.intercept_
B_1 = reg_model.coef_

print("B_0 = %f " % B_0)
print("B_1 = %f " % B_1)
print("Logit Function : f(x) = %f + %f*x" % (B_0, B_1))
print("Logistic Function : p(x) = 1/(1+exp(-f(x))) ")

color_code = matplotlib.colors.ListedColormap(['orange','green'])
scatter_plot = matplotlib.pyplot.scatter(X_j, P_j, c = y, cmap = color_code)
matplotlib.pyplot.legend(handles=scatter_plot.legend_elements()[0], labels=['0','1'], title = "Classes")
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel("X_j")
matplotlib.pyplot.ylabel("Y_j / N_j")
matplotlib.pyplot.show()

x_test = numpy.arange(0,6.01,0.1).reshape(-1,1)
logit_fx = B_0 + B_1 * x_test
logistic_px = 1/(1+numpy.exp(-1*logit_fx))

scatter_2 = matplotlib.pyplot.scatter(x, reg_model.predict(x))

l1 = matplotlib.pyplot.plot(x_test,logistic_px, color = 'orange',linewidth = 3, label = "Fitted logistic regression p(x)")
matplotlib.pyplot.legend(loc = "best")
matplotlib.pyplot.xlim(0,7)
matplotlib.pyplot.ylim(-0.1,1.1)
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel("X_j")
matplotlib.pyplot.ylabel("P_j")
matplotlib.pyplot.show()

print("The curve obtained from logistic regression appears to fit well to the given points.")

print("exp(B_1) : %f" % numpy.exp(B_1))
print("This number is the ODDS ratio")
print("This implies if we increase the dose by one unit, the fraction of insects dying (p/1-p) is expected to increase by 3.0667 times.")
