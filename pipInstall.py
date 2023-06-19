def checkAndInstall(package):
    # import pip
    import sys
    import subprocess

    try:
        # 없는 모듈 import시 에러 발생
        import package
    except:
        # pip 모듈 업그레이드
        subprocess.check_call([sys.executable,'-m', 'pip', 'install', '--upgrade', 'pip'])
        # 에러 발생한 모듈 설치
        subprocess.check_call([sys.executable,'-m', 'pip', 'install', '--upgrade', package])
        # 다시 import
        import package

def install(package, upgrade=True):
    import pip
    if hasattr(pip, 'main'):
        if upgrade:
            pip.main(['install','--upgrade',package])
        else:
            pip.main(['install',package])
    else:
        if upgrade:
            pip._internal.main(['install','--upgrade',package])
        else:
            pip._internal.main(['install',package])
    try:
        eval(f"import {package}")
    except ModuleNotFoundError:
        print("# Package name might be different. please check it")
    except Exception as e:
        print(e)