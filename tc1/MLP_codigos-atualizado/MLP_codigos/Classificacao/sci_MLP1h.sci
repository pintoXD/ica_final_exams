function [tree] = sci_MLP1h(tree)
// Generated by M2SCI
// Conversion function for Matlab MLP1h
// Input: tree = Matlab funcall tree
// Ouput: tree = Scilab equivalent for tree

tree=Funcall("exec",1,Rhs(tree.name),tree.lhs)
endfunction
