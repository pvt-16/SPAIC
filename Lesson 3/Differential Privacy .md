### Privacy
Learning about dataset without learning anything about the individual data itself.

### Differential Privacy
Cynthia Dwork - promise made by data holder to a data subject.
  Individual should not be affected, adversely or otherwise, by allowing their data to be used in any study or analysis, no matter what studies, data sets etc are available.
  
>  **Book** : The Algorithmic Foundations of Differential Privacy

### Anonymization doesn't always work?
Multiple, separate, anonymized datasets shows patterns that can be decoded using statistical techniques and publicly available data/ datasets.

Case Study 1: Netflix competition. Using IMDB movies and ratings database, one could map and deanonymize both users and the movies.

Case Study 2: https://www.forbes.com/sites/adamtanner/2013/04/25/harvard-professor-re-identifies-anonymous-volunteers-in-dna-study/#2dd1f7c792c9


### Canonical database
Canonical data models are a type of data model that aims to present data entities and relationships in the simplest possible form. 

*Simple database* - Here we create a database with one column that contains only 0 and 1 values. 
0 represents absence of a property and 1 represent presence. We are going with only one 1D tensor for now.

**How to define privacy in the context of this example database?**

If the query doesn't change even we remove someone from the database, then that person wasn't leaking any statistical information into the output of the query.

Let's first create a set of databases with one entry removed in each database.
Eg:If DB = [1,0,0,1,0,1...0,1]
    DB_1 = [0,0,1,0,1...0,1]  (Removed entry at 1st place '1')
    DB_2 = [1,0,1,0,1...0,1]  (Removed entry at 2nd place '0')
    and so on..
    
    
