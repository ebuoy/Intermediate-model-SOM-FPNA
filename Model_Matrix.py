def kohonen_model(n):
    
    M=[[0 for i in range (n)] for j in range (n)]
    
    V=[0,0,0,0,1]
    nulle=[0,0,0,0,0]
    
    mat_haut=[nulle]
    for i in range(1,4):
        mat_haut.append(V)
    mat_haut.append([0,1,1,1,0])
    
    mat_centre=[]    
    for i in range(4):
        mat_centre.append(V)
    mat_centre.append([1,1,1,1,0])
    
    mat_bas=[]
    for i in range(3):
        mat_bas.append(V)
    mat_bas.append([1,1,1,0,0])
    
    mat_gauche=[V,nulle,V,V,[1,0,1,1,0]]
    
    mat_droite=[V,V,nulle,V,[1,1,0,1,0]]
    
    
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                M[i][j]=[nulle,V,nulle,V,[0,1,0,1,0]]
            elif i == 0 and 0 <j < n-1:
                M[i][j]=mat_haut
            elif i == 0 and j==n-1:
                M[i][j]=[nulle,nulle,V,V,[0,0,1,1,0]]
            elif 0<i<n-1 and j == 0:
                M[i][j]=mat_gauche
            elif 0<i<n-1 and 0<j<n-1:
                M[i][j]=mat_centre
            elif 0<i<n-1 and j == n-1:
                M[i][j]=mat_droite
            elif i == n-1 and j == 0:
                M[i][j]=[V,V,nulle,nulle,[1,1,0,0,0]]
            elif i == n-1 and 0 <j < n-1:
                M[i][j]=mat_bas
            elif i == n-1 and j == n-1:
                M[i][j]=[V,nulle,V,nulle,[1,0,1,0,0]]
    
    return M
    
    
    
    

    
    
        