import {useState, useEffect} from 'react';

export const useFetch = (url) => {
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const [inProgress, setInProgress] = useState(false);

    useEffect(() => {
        const getData = async () => {
            // console.log("getData")
            try {
                setInProgress(true);
                const res = await fetch(url);
                const result = await res.json();
                tmp_array = new Array(result)
                setData(tmp_array);
            } catch (e){
                setError(e);
            } finally {
                setInProgress(false);
            }
        };
        getData();
        // console.log(data,"!!Data!!")
        // console.log(error,"!!Error!!")
        // console.log(inProgress,"!!inProgress!!")
    }, []);

    return { data, error, inProgress };
};