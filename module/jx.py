import sys
import gwas,gs,postGWAS,grm,pca,gformat

logo = r'''
       _                      __   __
      | |                     \ \ / /
      | | __ _ _ __  _   _ ___ \ V / 
  _   | |/ _` | '_ \| | | / __| > <  
 | |__| | (_| | | | | |_| \__ \/ . \ 
  \____/ \__,_|_| |_|\__,_|___/_/ \_\
'''
   
if __name__ == "__main__":
    module = dict(zip(['gwas','gs','postGWAS','grm','pca','gformat'],[gwas,gs,postGWAS,grm,pca,gformat]))
    extmodule = {}
    print(logo)
    if len(sys.argv)>1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            print(f'Usage: {sys.argv[0]} <module> [parameter]')
            print(f'''Modules: {' '.join(module.keys())}''')
        elif sys.argv[1] == '-v' or sys.argv[1] == '--version':
            print('JanusX v1.0.0')
        else:
            if sys.argv[1] in module.keys():
                module_name = sys.argv[1]
                sys.argv.remove(sys.argv[1])
                module[module_name].main()
            elif sys.argv[1] not in module.keys():
                print(f'Unkown module: {sys.argv[1]}')
                print(f'Usage: {sys.argv[0]} <module> [parameter]')
                print(f'''Modules: {' '.join(module.keys())}''')
    else:
        print(f'Usage: {sys.argv[0]} <module> [parameter]')
        print(f'''Modules: {' '.join(module.keys())}''')