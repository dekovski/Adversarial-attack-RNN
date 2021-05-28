# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:03:48 2020

@author: shank
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height ),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


matplotlib.rcParams['figure.dpi'] = 300      

labels = ['0.01', '0.05', '0.1']
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()

grad = np.zeros(len(labels))
fp = np.zeros(len(labels))
fpd = np.zeros(len(labels))

t_grad = np.zeros(len(labels))
t_fp = np.zeros(len(labels))
t_fpd = np.zeros(len(labels))

for k,eps in enumerate(labels):
    t_grad[k],grad[k] = np.load('data/compare/grad_class_0_' + eps + '.npy')
    t_fp[k],fp[k] = np.load('data/compare/fp_class_0_' + eps + '.npy')
    t_fpd[k],fpd[k] = np.load('data/compare/fp_dyn_class_0_' + eps + '.npy')

grad = np.round(0.1*grad,1)
fp = np.round(0.1*fp,1)
fpd = np.round(0.1*fpd,1)

rects1 = ax.bar(x - width, grad , 0.8*width, label='Gradient', color = 'darkgreen')
rects2 = ax.bar(x , fp, 0.8*width, label='Fixed-point', color = 'greenyellow')
rects3 = ax.bar(x + width, fpd,0.8* width, label='Dynamic fixed-point', color = 'mediumspringgreen')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Attack success rate (%)')
ax.set_title('Attack on class 0 nominal signals')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_xlabel('Perturbation size')
ax.set_aspect(0.0125)
plt.ylim(0,120)
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
#plt.grid(True)
plt.show()

#############

labels = ['0.2', '0.3', '0.4']
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
ax.set_aspect(0.08)
grad = np.zeros(len(labels))
fp = np.zeros(len(labels))
fpd = np.zeros(len(labels))

t_grad = np.zeros(len(labels))
t_fp = np.zeros(len(labels))
t_fpd = np.zeros(len(labels))

for k,eps in enumerate(labels):
    t_grad[k],grad[k] = np.load('data/compare/grad_class_1_' + eps + '.npy')
    t_fp[k],fp[k] = np.load('data/compare/fp_class_1_' + eps + '.npy')
    t_fpd[k],fpd[k] = np.load('data/compare/fp_dyn_class_1_' + eps + '.npy')

grad = np.round(0.1*grad,1)
fp = np.round(0.1*fp,1)
fpd = np.round(0.1*fpd,1)

rects1 = ax.bar(x - width, grad , 0.8*width, label='Gradient',color='indigo')
rects2 = ax.bar(x , fp, 0.8*width, label='Fixed-point', color = 'darkorchid')
rects3 = ax.bar(x + width, fpd, 0.8*width, label='Dynamic fixed-point', color = 'plum')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Attack success rate (%)')
ax.set_title('Attack on class 1 nominal signals')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0,18)
ax.set_xlabel('Perturbation size')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
#plt.grid(True)
plt.show()

#############


fig, ax = plt.subplots()

# Example data
labels = ['Gradient', 'Fixed-Point', 'Dynamic F-P']
y_pos = np.arange(len(labels))
rt = np.around([np.sum(t_grad)/3000,np.sum(t_fp)/3000,np.sum(t_fpd)/3000],2)


ax.barh(y_pos, rt, 0.6, align='center', color='orangered')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('seconds')
ax.set_title('Runtime comparison')
ax.set_aspect(0.2)

for k in range(3):
    ax.annotate( rt[k],
                        xy=(rt[k]+0.08, y_pos[k] + 0.15 ),
                        xytext=(0,0),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
plt.xlim(0,2)
plt.show()













