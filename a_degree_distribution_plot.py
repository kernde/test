import numpy
import numpy as np
import matplotlib.pyplot as plt # matplotlib.use('TkAgg')
from .graphs_loading import load_company_graph
import powerlaw

G = load_company_graph(extended=False,output_type='NetworkX')
Ge = load_company_graph(extended=True,output_type='NetworkX')

degree = np.array([d for n, d in G.degree()])
degreee = np.array([d for n, d in Ge.degree()])

dg = numpy.unique(degree, return_counts=True)
dge = numpy.unique(degreee, return_counts=True)

#results.power_law.plot_pdf(color= 'b',linestyle='--',label='fit ccdf')
plt.scatter(dg[0], dg[1]/sum(dg[1]), label="Company") # div by sum for freq instead of counts
plt.scatter(dge[0], dge[1]/sum(dge[1]), label="Company Extended")
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Degree k")
plt.ylabel("Density P(k)")
plt.legend(loc="upper right")


import tikzplotlib
tikzplotlib.save("output_files/degree_distribution.tex")

#edges, hist = powerlaw.pdf(degree)
##bin_centers = (edges[1:]+edges[:-1])/2.0
#plt.loglog(edges, hist)