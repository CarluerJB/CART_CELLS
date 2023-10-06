# ===============================
# AUTHOR     : CARLUER Jean-Baptiste
# CREATE DATE     : 2022-2023
# PURPOSE     : Thesis in BioInformatics
# SPECIAL NOTES: Setup and download evertything to be able to run the pipeline
# ===============================
# Change History:
#                  
# # =================================

PS3='In order to analyse GO term in your data, you must download the most current GO, Please enter your choice: '
options=("GO" "GO Slim" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "GO")
            wget http://current.geneontology.org/ontology/go-basic.obo
            break
            ;;
        "GO Slim")
            wget http://current.geneontology.org/ontology/subsets/goslim_generic.obo
            break
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

PS3='In order to analyse GO term in your data, you must download the goatools from the github of tanghaibao: '
options=("Accept" "Decline")
select opt in "${options[@]}"
do
    case $opt in
        "Accept")
            svn checkout https://github.com/tanghaibao/goatools/trunk/goatools
            break
            ;;
        "Decline")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

PS3='In order to estimate the most important metrics of the model, you must download the AraNetBench evaluation method from the github of Oceane Cassan: '
options=("Accept" "Decline")
select opt in "${options[@]}"
do
    case $opt in
        "Accept")
            svn export https://github.com/OceaneCsn/AraNetBench.git/trunk/R/evaluateNetwork.R
            svn export https://github.com/OceaneCsn/AraNetBench.git/trunk/data/validated_edges.rda
            mv evaluateNetwork.R NetworkEvaluation/
            mv validated_edges.rda NetworkEvaluation/
            break
            ;;
        "Decline")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

PS3='In order to execute the full pipeline some python package are needed, install in a conda environment ?: '
options=("Accept" "Decline")
select opt in "${options[@]}"
do
    case $opt in
        "Accept")
            conda env create -f environment.yml
            conda activate CART_CELLS
            break
            ;;
        "Decline")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done
