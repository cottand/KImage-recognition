# KImage Recognition - CURRENTLY WIP

### Goal

The aim of this project is mainly to learn by building a small but easily expandable fully
featured CNN.


### Design decisions

 - **Kotlin instead of Python?** The main focus here is to learn the workings of a CNN without
  relying on any libraries (TensorFlow?) to do the work for me. Using Kotlin makes me need to
   consider types and stops me from writing stupid things.
   
 - **NumPy bindings instead of a well established data science Kotlin lib?** While this is not
  python, I still wanted the code to be close, and in particular readable to the Python user.
   Kotlin + NumPy is a good idea since preserving the API across languages will keep the code
   familiar.
   
 - **This could actually be a lot shorter**. I know, I am learning. I wanted to abstract classes
  away as is common in software engineering, since that was my background when I started writing
   this. The resulting modularity does allow for easy extensibility.