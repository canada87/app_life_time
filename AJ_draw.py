import matplotlib.pyplot as plt
#import csv
import numpy as np
from matplotlib import cm

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

palette = ["#1F77B4","#FF7F0E","#2CA02C", "#00A3E0", '#4943cf', '#1eeca8', '#e52761', '#490b04', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b',
                   '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff',
                   '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4']


#workflow:
#    import_data
#    nuova_fig
#    titoli
#    dati
#    range_plot
#    legenda
#    testo_su_figura
#    colora_assi
#    salva_graf
#    porta_a_finestra

class disegna:
    """
    class to draw data quickly
    order of the workflow:

    1. import_data
    2. nuova_fig (mandatory)
    3. titoli (mandatory)
    4. dati (mandatory)
    5. range_plot
    6. legenda
    7. testo_su_figura
    8. colora_assi
    9. salva_graf
    10. salva_dati
    11. porta_a_finestra (mandatory)
    12. aggiusta_la_finestra (if you use a different show function)

    """

#############################################
    def import_data(self, name_file, separator=' ', single_file = 'yes', name_file_y = ''):
        """
        :param name_file: name of the file where the data are
        :param separator: useless parameter
        :param single_file: 'yes' or 'no', if the X and Y are splitted onto 2 files select 'yes'
        :param name_file_y: name of the file with the Y data if single_file='yes'
        """
        global dat_x, dat_y
        dat_x = []
        dat_y = []
        skip_line=0
        if single_file=='yes':
            dat_x = np.loadtxt(fname = name_file, skiprows=skip_line, usecols=[0])
            dat_y = np.loadtxt(fname = name_file, skiprows=skip_line, usecols=[1])

#            with open(name_file,'r') as csvfile:
#                plots = csv.reader(csvfile)#, delimiter=separator)
#                for row in plots:
#                    dat_x.append(int(row[0]))
#                    dat_y.append(int(row[1]))

        else:
            plots_x = open(name_file, 'r')
            plots_y = open(name_file_y, 'r')

            for element in plots_x:
                dat_x.append(float(element))
            for element2 in plots_y:
                dat_y.append(float(element2))

        return dat_x, dat_y

#            with open(name_file,'r') as csvfile:
#                plots = csv.reader(csvfile)
#                print(plots)
#
#            with open(name_file_y,'r') as csvfile:
#                dat_y = csv.reader(csvfile)

#############################################
    def nuova_fig(self, indice_fig=1, indice_subplot = 111, width = 6.4, height = 4.8, plot_dimension = None):
        """
        open a new canvas

        :param indice_fig: number of the canvas
        :param indice_fig: number of the position for the subplot into the canvas
        """
        global fig, ax1
        fig = plt.figure(indice_fig, figsize=(width,height))

        # if plot_dimension == '2d':
        #     ax1 = plt.subplot(indice_subplot)
        # else:
        #     ax1 = fig.gca(projection='3d')
        ax1 = plt.subplot(indice_subplot, projection=plot_dimension)#projection = '3d' per il graph 3d
        return fig, ax1


#############################################
    def titoli(self, titolo="titolo", xtag="X", ytag="Y", griglia=1, sub_plot_num=1, labelsize = 13):
        """
        label the image with titles

        :param titolo: title of the image
        :param xtag: name of the X axe
        :param ytag: name of the Y axe
        :param griglia: with 1 doesn't appear the grid, with any other numbers it does
        :param sub_plot_num: indicates in which subplot the title goes
        """
        if sub_plot_num==1:
#            global ax1
            ax1.set_title(titolo, y = 1.08)
            ax1.set_xlabel(xtag, fontsize = labelsize)
            ax1.set_ylabel(ytag, fontsize = labelsize)
            ax1.tick_params(axis='both', which = 'major', labelsize=labelsize)
            if griglia != 1:
                ax1.grid(True)
        else:
            global ax2
            ax2 = ax1.twinx()
            ax2.set_title(titolo)
            ax2.set_ylabel(ytag)
            ax2.tick_params(axis='both', which = 'major', labelsize=labelsize)
            if griglia != 1:
                ax2.grid(True)

#############################################
    def dati(self, x, y, x_error = 0, y_error=0, colore="#00A3E0", descrizione="", sub_plot_num=1, scat_plot = 'line', larghezza_riga = 1, z = None, delay = -3, width = 3, layer = None):
        """
        import data on the canvas

        :param x: array with the X data, if empty the loaded data with import_data are used
        :param y: array with the Y data, if empty the loaded data with import_data are used
        :param colore: color used for the data visualization (string)
        :param descrizione: how to label the data for the legend
        :param sub_plot_num: indicates in which subplot the title goes
        :param scat_plot: 'line' or 'scat' or 'err' or 'cmap' or '3D', if 'line' the data are represent with line, if 'scat' as scatter plot, if 'err' as scatter with error bars, if 'cmap' as colormap (require the z parameter), '3D' as tridimensional plot
        :param larghezza_riga: represent the thickness of the line if scat_plot=1
        :param z: array with the Z datta, has to be a matrix with Z[len(x),len(y)]
        """

        if sub_plot_num==1:
            ax1.tick_params(axis='x', direction = 'in')
            ax1.tick_params(axis='y', direction = 'in')
            if scat_plot == 'scat':
                ax1.scatter(x,y,color=colore, label=descrizione, s = larghezza_riga, zorder = layer)
            if scat_plot == 'line':
                ax1.plot(x,y,color=colore, label=descrizione, linewidth = larghezza_riga)
            if scat_plot == 'err':
                ax1.errorbar(x,y,xerr=x_error, yerr=y_error, label=descrizione, color= colore, linestyle = '')
            if scat_plot == 'cmap':
                z_min, z_max = np.abs(z).min(), np.abs(z).max()
                c = ax1.pcolormesh(x, y, z, cmap='plasma', vmin=z_min, vmax=z_max)
                fig.colorbar(c, ax=ax1)
            if scat_plot == '3D':
                ax1.plot_surface(x,y,z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            if scat_plot == '3D_wire':
                ax1.plot_wireframe(x,y,z, color=colore)#, label=descrizione, s = larghezza_riga)
            if scat_plot == 'hist':
                ax1.hist(x, bins = y, color = colore)
            if scat_plot == 'bar':
                ax1.bar(x[:] + delay/2, y, width, label=descrizione)
                ax1.set_xticks(x)
                if z != None:
                    ax1.set_xticklabels(z, rotation = 45)

        else:
            ax2.tick_params(axis='y', direction = 'in')
            if scat_plot == 'scat':
                ax2.scatter(x,y,color=colore, label=descrizione, s = 0.5)
            if scat_plot == 'line':
                ax2.plot(x,y,color=colore, label=descrizione, linewidth = larghezza_riga)
            if scat_plot == 'err':
                ax2.errorbar(x,y,xerr=x_error, yerr=y_error, label=descrizione, color= colore, linestyle = '')

#############################################
    def testo_su_figura(self, testo='testo', coordX = 0, coordY = 0, dimensione_testo = 15, colore = 'black', con_freccia = 'no', coordX_freccia = 1, coordY_freccia = 1):
        """
        overimpose some text on the image

        :param testo: (string) insert the text on the image
        :param coordX: X coordinate where the text is inserted
        :param coordY: Y coordinate where the text is inserted
        :param dimensione_testo: font size of the text
        :param colore: (string) color of the text
        :param con_freccia: 'yes' or 'no', an arrow is inserted if 'yes'
        :param coordX_freccia: X coordinate where the arrow is inserted
        :param coordY_freccia: Y coordinate where the arrow is inserted
        """
        if con_freccia == 'no':
            font_dict = {'family': 'serif',
                         'color': colore,
                         'size':dimensione_testo}
            ax1.text(coordX, coordY, testo, fontdict = font_dict)
        else:
            ax1.annotate(testo, (coordX_freccia, coordY_freccia), xytext = (coordX, coordY),
                         arrowprops = dict(facecolor = colore, color = colore), size = dimensione_testo)

#############################################
    def colora_assi(self, colore_asse_sx='black', colore_asse_dx='red'):
        """
        modify the color of the axes

        :param colore_asse_sx: (string) select the color for the left axe
        :param colore_asse_dx: (string) select the color for the right axe
        """
        ax1.yaxis.label.set_color(colore_asse_sx)
        ax1.tick_params(axis='y', colors=colore_asse_sx)
        ax2.spines['left'].set_color(colore_asse_sx)
        ax2.yaxis.label.set_color(colore_asse_dx)
        ax2.spines['right'].set_color(colore_asse_dx)
        ax2.tick_params(axis='y', colors=colore_asse_dx)

#############################################
    def range_plot(self, bottomX = None, topX = None, bottomY = None, topY = None, sub_plot_num=1):
        """
        change the range of the plot

        :param bottomX: lower limit on the X axe
        :param topX: upper limit on the X axe
        :param bottomY: lower limit on the Y axe
        :param topY: upper limit on the Y axe
        :param sub_plot_num: indicates which subplot has a custom range
        """
        if sub_plot_num==1:
            ax1.set_xlim(bottomX, topX)
            ax1.set_ylim(bottomY, topY)
        else:
            ax2.set_ylim(bottomY, topY)
            ax2.set_xlim(bottomX, topX)

#############################################
    def legenda(self, sub_plot_num=1):
        """
        insert the legend on the image

        :param sub_plot_num: indicates which subplot has the legend
        """
        if sub_plot_num==1:
            h1, l1 = ax1.get_legend_handles_labels()
            htot = h1
            ltot = l1
        else:
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            htot = h1 + h2
            ltot = l1 + l2
        ax1.legend(htot, ltot, loc=1).get_frame().set_alpha(0.4)
        # ax1.legend(htot, ltot, loc=9, bbox_to_anchor=(0,0.99,1,.102), ncol=3, borderaxespad=0).get_frame().set_alpha(0.4)


#############################################
    def salva_graf(self,titolo="titolo"):
        """
        save the image plotted into an image file

        :param titolo: title of the output file
        """
        fig.tight_layout()
        plt.savefig(titolo+'.png', dpi=300, transparent = True)

#############################################
    def porta_a_finestra(self, chiudi = 0):
        """
        show the plot into a canvas
        """
        if chiudi == 0:
            fig.tight_layout()
            plt.show()
        else:
            plt.close()

#############################################
    def aggiusta_la_finestra(self):
        """
        fit the plot into a canvas
        """
        fig.tight_layout()

#############################################
    def salva_dati(self, x = 'empty', y = 'empty', with_error='no', x_error = 0, y_error = 0, nomefile = 'dati_new.txt'):
        """
        save the data plotted into a txt file

        :param x: array with the X data, if empty the loaded data with import_data are used
        :param y: array with the Y data, if empty the loaded data with import_data are used
        :param with_error: 'yes' or 'no', if 'yes' you can enter the error bar values
        :param x_error: is the array of the error bar values on X
        :param y_error: is the array of the error bar values on Y
        :param nomefile: (string) name of the output file, with the extension
        """
        if with_error == 'no':
            if x=='empty' or y=='empty':
                x = dat_x
                y = dat_y
            dati = np.array([x, y])
            dati = dati.transpose()
            np.savetxt(nomefile, dati, delimiter=' ', newline='\n', header='', footer='')
        else:
            dati = np.array([x, y, x_error, y_error])
            dati = dati.transpose()
            np.savetxt(nomefile, dati, delimiter=' ', newline='\n', header='', footer='')
