## Dhund (WIP)


Dhund is a semantic search engine (that internally uses vector embeddings) that is optimized for the multi-modal use case. 
As of 2024, Dhund is focussed on the use case of an end-to-end e-commerce semantic search engine. 


### Why dhund? Why not use an off-the-shelf vector DB?
Because it isn't sufficient to have storage for your embeddings.
How do you create the embeddings? 
How do you then store them in an optimal format?
How do you tune the performance of your search engine?

The performance of all the vector DBs, in terms of relevance, is, at the end of the day, dependent on the accuracy of your embeddings and their relevance to your actual products. 
Dhund proposes a system where an embeddings generator model is first fine-tuned on your dataset (You will not need to fine-tune this model each time a new product is added) 
which then will be used in the vector DB. 

Dhund offers all the features required for this end-to-end, so you as a developer just need to provide a source of your Products data (in CSV for example) and Dhund creates a 
**semantic search engine** for you!

### Why build in Rust and not in Python?
Because I like rust!

