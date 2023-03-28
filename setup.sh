
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

PS3='In order to analyse GO term in your data, you must download the goatools from the github of tanghaibao'
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