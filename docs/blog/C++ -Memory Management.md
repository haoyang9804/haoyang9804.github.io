# C++ -Memory Management

I have been bothered with one of the C++ memory management recipe, `std::move`, for a long time. But these days, I got some inspiration from the language design of Rust for managing memory, with which I try to understand the C++ counterpart.

---

Here is a C++ boilerplate code from [here](https://www.cprogramming.com/c++11/rvalue-references-and-move-semantics-in-c++11.html) with slight modification.

```c++
class ArrayWrapper
{
public:
    // default constructor produces a moderately sized array
    ArrayWrapper ()
        : _p_vals( new int[ 64 ] )
        , _size( 64 )
    {}
 
    ArrayWrapper (int n, string name)
        : _p_vals( new int[ n ] )
        , _size( n )
        , _name(name)
    {
    }
 
    // move constructor
    ArrayWrapper (ArrayWrapper&& other)
        : _p_vals( other._p_vals  )
        , _size( other._size )
        , _name( other._name )
    {
        cout << "move " << this->_name << endl;
        other._p_vals = NULL;
        other._size = 0;
    }
 
    // copy constructor
    ArrayWrapper (const ArrayWrapper& other)
        : _p_vals( new int[ other._size  ] )
        , _size( other._size )
        , _name( other._name )
    {
        cout << "copy " << this->_name << endl;
        for ( int i = 0; i < _size; ++i )
        {
            _p_vals[ i ] = other._p_vals[ i ];
        }
    }

    ~ArrayWrapper ()
    {
        cout << "delete " << this->_name << endl;
        delete [] _p_vals;
    }

    const int& size_() {
      return this->_size;
    }

    void rename(string name) {
      this->_name = name;
    }

private:
    int *_p_vals;
    int _size;
    string _name;
};
```

To test the copy constructor, I write the following code, which is the one I always wrote before.

```c++
int main() {
  ArrayWrapper aw1(12, "aw1");
  ArrayWrapper aw2(aw1); aw2.rename("aw2");
}
```

The output is trivial and intuitive:

```
copy aw1
delete aw2
delete aw1
```

`ArrayWrapper aw3(aw1)` invokes the copy constructor and thus prints `copy aw1` on terminal.

The introduction of `std::move` here would invoke the move instructor. To prove it, I rewrite the `main` function:

```c++
int main() {
  ArrayWrapper aw1(12, "aw1");
  ArrayWrapper aw2(aw1); aw2.rename("aw2");
  ArrayWrapper aw3(std::move(aw1)); aw3.rename("aw3");
}
```

The output shows that the move constructor has been invoked.

```
copy aw1
move aw1
delete aw3
delete aw2
delete aw1
```

One observation is in our move constructor, the re-assignment of the rvalue-reference is important to promise no memory problem in execution.

```c++
// move constructor
    ArrayWrapper (ArrayWrapper&& other)
        : _p_vals( other._p_vals  )
        , _size( other._size )
        , _name( other._name )
    {
        cout << "move " << this->_name << endl;
        other._p_vals = NULL;
        other._size = 0;
    }
```

If I comment out

```c++
other._p_vals = NULL;
other._size = 0;
```

, then the process of deleting `aw` by destructor will cause a malloc fault in execution. This is because the content owned by `aw1` has been moved to `aw3`, and thus destruct `aw1` will try to release some unknown memory.

---

In continuing with this trivial experiment, I also encountered **R**eturn **V**alue **O**ptimization.

Concretely speaking, I add a new function to return a `ArrayWrapper` instance immediately after creating it.

```c++
ArrayWrapper createAW_() {
  ArrayWrapper aw_3(15, "aw_3");
  return aw_3;  
}
```

And I call it in `main` function to expect for an invokation of move constructor. But sadly, I failed.

```c++
ArrayWrapper aw4(createAW_()); aw4.rename("aw4");
```

The output is `delete aw4`, but without printing ` move aw4`. This is because the compiler implicitly optimizes it at compile time (RVO).

But what if the compiler has no idea how to optimize it at compile time?

I rewrite `createAW_` by adding an `if` statement to complicate the control flow. This modification requires knowledge of the value of `x` during runtime to do code analysis and then optimization. The code is as below:

```c++
ArrayWrapper createAW(int x) {
  ArrayWrapper aw_1(x, "aw_1");
  ArrayWrapper aw_2(13, "aw_2");
  if (aw_1.size_() > aw_2.size_())
    return aw_1;
  return aw_2;
}
```

And I add a new `ArrayWrapper` instance in `main` function:

```c++
ArrayWrapper aw5(createAW(12)); aw5.rename("aw5");
```

The output shows that move constructor has been invoked again:

```
move aw_2
delete aw_2
delete aw_1
delete aw5
```

