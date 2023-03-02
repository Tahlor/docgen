import torch

def try_try_again_factory(tries=3,
                          debug=False,
                          early_fail_fallback_function=None):
    def try_try_again(func):
        def try_again(*args,**kwargs):
            i = 0
            if debug:
                return func(*args,**kwargs)

            while True:
                try:
                    i+=1
                    result = func(*args, **kwargs)
                    return result

                except KeyboardInterrupt as e:
                    if early_fail_fallback_function:
                        early_fail_fallback_function()
                except Exception as e:
                    print(f"ERROR {i} {e}")
                    torch.cuda.empty_cache()
                    if i >= tries:
                        if early_fail_fallback_function:
                            early_fail_fallback_function()
                        return
        return try_again
    return try_try_again