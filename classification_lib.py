from AJ_draw import disegna as ds
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.evaluate import confusion_matrix
import seaborn as sns

import streamlit as st
class marketing:
    def score_accuracy_recall(self, y_pred_matrix, y_true, verbose = 1):
        """evaluate accuracy recall and plot the confusion matrix for base learners and ensemble.
        :param y_true: array with the targets, pandas (it is NOT onehot encoded)
        :param y_pred_matrix: pandas matrix with the predicted class for each model as column, the prediction comes in the same format of the target (it is NOT onehot encoded)
        :param verbose: 0 or 1, with 1 the confusion matrix is ploted
        :return: accurary_real.shape[0] = y_pred_matrix.shape[0], accurary_real.shape[1] = (accuracy_tot, recall_average, precision_average), pandas matrix with the models and some scores
        """
        # labels = np.sort(y_true.unique())
        accurary_real = pd.DataFrame()
        for col in y_pred_matrix.columns:
            cm = confusion_matrix(y_target=y_true, y_predicted=y_pred_matrix[col], binary=False)
            prob_cm =  np.zeros_like(cm).astype(float)
            # matrix with the predicted vs true values
            df_cm = pd.DataFrame(cm)

            # df_cm.columns = ['true '+str(i) for i in range(df_cm.shape[1])]
            # df_cm.index = ['predict '+str(i) for i in range(df_cm.shape[0])]

            tot_true_val = []
            tot_predic_val = []
            recall = []
            precision = []

            #calculate the recall and precision for each class separatly
            for i in range(cm.shape[0]):
                #create the initial matrix with the percentage of cases in each class
                #il 100% viene fatto sommando assieme tutti i casi in una colonna che rappresentano il totale dei casi reali
                prob_cm[i,:] = cm[i,:]/cm[i,:].sum()

                # la totalita dei casi reali per ciascuna classe viene calcolata sommando assieme tutti i casi in ciascuna colonna
                tot_true_val.append(cm[i,:].sum())# rapresents the TP (correct predicted data that are on the diagonal) + FN (tutti quelli che fanno parte di una classe ma che non tutti sono stati erroneamente messi qui)
                # la totalita dei casi predetti per ciascuna classe viene calcolata sommando assieme tutti i casi in ciascuna riga
                tot_predic_val.append(cm[:,i].sum())# rapresents the TP (correct predicted data that are on the diagonal) + FP (tutti quelli che sono statti predetti appartenere ad una classe anche se non tutti ne fanno parte)

                #calcolo del recupero (sensibilita) = TP/(TP+FN)
                if tot_true_val[i] == 0:
                    recall.append(0)
                else:
                    recall.append(round(cm[i,i]/tot_true_val[i],2))

                #calcolo della precisione = TP/(TP+FP)
                if tot_predic_val[i] == 0:
                    precision.append(0)
                else:
                    precision.append(round(cm[i,i]/tot_predic_val[i],2))

            #add to the main matrix the total true values for each classes
            df_cm['tot true'] = tot_true_val

            #add recall of each class to the matrix
            df_recall = pd.DataFrame(recall, columns = ['recall'])
            df_cm = pd.concat([df_cm, df_recall], axis = 1)

            #vector with the prediction of each class
            df_predict = pd.DataFrame(tot_predic_val, columns = ['tot predict']).T
            # calcola il numero totale di campioni sommando assieme tutti i campioni che realmente appartengono ad ciascuna classe
            tot_samples = df_cm['tot true'].sum()
            #aggiunge al vettore dei campioni che realmente appartengono a ciuscuna scalla il numero totale dei campioni
            df_predict['tot true'] = tot_samples
            #avegare recall from each class
            df_predict['recall'] = round(df_cm['recall'].sum()/len(recall),2)
            #add at the matrix the total prediction of each class, the total number of cases and the average recall
            df_cm = pd.concat([df_cm, df_predict])

            #create a matrix with the precision for each classes
            df_precision = pd.DataFrame(precision, columns = ['precision']).T
            #add to the matrix the average precision
            df_precision['tot true'] = round(df_precision.loc['precision'].sum()/len(precision),2)

            accuracy_tot = 0
            accuracy_average_tot = 0

            #viene calcolata l'accuratezza (TP+TN)/(TN+TP+FN+TP), modo complicato per dire tutto quello che e' stato predetto correttamente diviso il totale dei casi
            for i in range(cm.shape[0]):
                accuracy_tot = cm[i,i] + accuracy_tot# TP + TN che rappresentano tutti gli indovinati correttamente
                accuracy_average_tot = prob_cm[i,i]+accuracy_average_tot# TP + TN ma in precentuale
            accuracy_average_tot = accuracy_average_tot/len(cm[:,0])#TP+TN/nunmero_classi, rappresenta l'accuratezza media tra tutte le classi
            accuracy_tot = accuracy_tot/len(y_true)#somma di tutti i casi correttamente indovinati (TP+TN) diviso il numero di casi totali, rappresenta l'accuratezza totale

            #viene aggiunta al vettore recall il valore accuratezza totale
            df_precision['recall'] = round(accuracy_tot,2)
            #viene messo nella matrice principale il vettore con la precisione per ogni classe, la precisione media e l'accuratezza totale
            df_cm= pd.concat([df_cm, df_precision])

            temp_index = df_cm.index.tolist()[-2:]
            temp_index =  ['True '+str(i) for i in range(df_cm.shape[0]-2)] + temp_index
            df_cm.index = temp_index
            temp_col = df_cm.columns.tolist()[-2:]
            temp_col =  ['Predict '+str(i) for i in range(df_cm.shape[0]-2)] + temp_col
            df_cm.columns = temp_col

            accurary_real[col] = [accuracy_tot, accuracy_average_tot, df_cm['tot true']['precision'], df_cm['recall']['tot predict']]
            if verbose == 1:
                # fig, ax = plot_confusion_matrix(conf_mat=cm, show_normed=True, show_absolute=True, colorbar=True, class_names=labels)
                # plt.show()
                # sns.heatmap(df_cm, annot=True, annot_kws={'size': 8}, cmap=plt.cm.Blues, vmax=tot_samples, vmin=0, square=True, linewidths=0.5, linecolor="black")
                # sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, vmax=tot_samples, vmin=0, square=True, linewidths=0.5, linecolor="black", cbar=False)
                # plt.show()

                df_cm_ar = df_cm.to_numpy()
                df_cm_ar_temp = df_cm_ar.copy()
                fig, ax = plt.subplots()
                df_cm_ar_temp[-2:, -2:] = -500
                df_cm_ar_temp[:-2, -2:-1] = -200
                df_cm_ar_temp[-2:-1, :-2] = -200
                df_cm_ar_temp[:-2, -1] = -300
                df_cm_ar_temp[-1, :-2] = -300

                ax.imshow(df_cm_ar_temp, cmap=plt.get_cmap('cool'))
                ax.set_xticks(np.arange(df_cm_ar.shape[1]))
                ax.set_yticks(np.arange(df_cm_ar.shape[0]))
                ax.set_xticklabels(df_cm.columns.tolist())
                ax.set_yticklabels(df_cm.index.tolist())
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                for i in range(df_cm_ar.shape[0]):
                    for j in range(df_cm_ar.shape[1]):
                        ax.text(j, i, df_cm_ar[i, j], ha="center", va="center", color="black")

                ax.text(df_cm_ar.shape[1]-1, df_cm_ar.shape[0]-1-0.3, 'Ave Acc tot', ha="center", va="center", color="black")
                ax.text(df_cm_ar.shape[1]-1, df_cm_ar.shape[0]-2-0.3, 'Ave Recall', ha="center", va="center", color="black")
                ax.text(df_cm_ar.shape[1]-2, df_cm_ar.shape[0]-2-0.3, 'Tot Counts', ha="center", va="center", color="black")
                ax.text(df_cm_ar.shape[1]-2, df_cm_ar.shape[0]-1-0.3, 'Ave Prec', ha="center", va="center", color="black")

                ax.set_title(col)
                fig.tight_layout()
                st.pyplot()

        accurary_real = accurary_real.T
        accurary_real.columns = ['accuracy_tot', 'accuracy_average', 'precision_average', 'recall_average']
        return accurary_real

    def correlation_matrix(self, data, col_y, corr_value = 0.95, corr_value_w_targhet = 0.95, plot_matr = 'yes'):
        """
        'col_y' represent the target column
        """
        corr = data.corr().abs()

        if plot_matr == 'yes':
            ds().nuova_fig(7, height =8, width =10)
            ds().titoli(titolo="Correlation matrix")
            sns.heatmap(corr[(corr >= 0.5)], cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1, annot=True, annot_kws={"size": 8}, square=True, linecolor="black");
            ds().aggiusta_la_finestra()
            st.pyplot()

        corr_with_target = corr[col_y]#correlation with the target
        relevant_feature_with_target = corr_with_target[corr_with_target < corr_value_w_targhet].sort_values(ascending = False).reset_index()

        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))#select upper triangle of correlation matrix
        correlation_between_parameters = [column for column in upper.columns if any(upper[column] > corr_value)]
        return relevant_feature_with_target, correlation_between_parameters
