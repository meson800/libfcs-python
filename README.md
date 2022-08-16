# libfcs-python
Python bindings for libfcs

https://ro-che.info/articles/2015-10-26-static-linking-ghc
https://www.hobson.space/posts/haskell-foreign-library/


TODO: try using nested offsets. E.g. if you are trying to find the pointer
in order to free it, use
#{offset FCSFile name.buffer}