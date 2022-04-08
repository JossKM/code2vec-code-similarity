int faaa(int n) {
    if (n == 1) {
        return 1; 
    } else {
        return n * f(n-2);
    }
}