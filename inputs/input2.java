int q(int naaa) {
    if (naaa == 1) {
        return 1; 
    } else {
        return naaa * f(naaa-1);
    }
}