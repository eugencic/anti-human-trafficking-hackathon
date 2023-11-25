import React, {useEffect, useState} from "react";
import Projects from "../admin/profile/components/Projects";
import General from "../admin/profile/components/General";
import {Grid} from "@chakra-ui/react";
import {useParams} from "react-router-dom";
import axios from "redaxios";

function Region() {
    let {regionId} = useParams();

    const [projects, setProjects] = useState([])
    const [selectedProject, setSelectedProject] = useState(0)

    useEffect(() => {
        axios.get(`http://localhost:5000/api/region/${regionId}/2023`).then((response) => {
            setProjects(response.data)
        });
    }, [regionId]);

    return (
        <>
            <Grid
                mb='20px'
                marginTop="10%"
                templateColumns={{
                    base: "1fr",
                    lg: "repeat(2, 1fr)",
                    "2xl": "1fr 2fr",
                }}
                templateRows={{
                    base: "1fr",
                    lg: "repeat(2, 1fr)",
                    "2xl": "1fr",
                }}
                gap={{ base: "20px", xl: "20px" }}>
                <Projects
                    gridArea='1 / 2 / 2 / 2'
                    projects={projects}
                    setSelectedProject={setSelectedProject}
                    name='Adela Parkson'
                    job='Product Designer'
                    posts='17'
                    regionId={regionId}
                    followers='9.7k'
                    following='274'
                />
                <General
                    gridArea={{ base: "2 / 1 / 3 / 2", lg: "1 / 2 / 2 / 3" }}
                    minH='365px'
                    selectedProject={projects[selectedProject]}
                    pe='20px'
                />
            </Grid>
        </>

    );
}

export default Region;
