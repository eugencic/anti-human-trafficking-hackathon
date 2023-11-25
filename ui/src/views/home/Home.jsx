import './Home.css';
import React, { useRef } from 'react';
import MySvg from '../../assets/map.svg';
import { ReactSVG } from 'react-svg';
import { useHistory } from 'react-router-dom';
import {Link} from "@chakra-ui/react";

function Home() {
    const tooltipRef = useRef();
    const history = useHistory();

    const handlePathClick = (id) => { 
        history.push(`/admin/regions/${id}`);
    };

    function mouseEntered(e, data) {
        const target = e.target;

        if (target.nodeName === 'path') {
            target.style.opacity = 0.6;
            tooltipRef.current.style.transform = `translate(${e.offsetX + 200}px, ${e.offsetY + 50}px)`;

            tooltipRef.current.innerHTML = `
        <ul>
            <li><b>Raionul ${data.name}</b></li>
            <li>Current: ${data.current}</li>
            <li>Prediction: ${data.prediction}</li>
        </ul>`;
        }
    }

    function mouseGone(e) {
        const target = e.target;

        if (target.nodeName === 'path') {
            target.style.opacity = 1;
            tooltipRef.current.innerHTML = '';
        }
    }

    const attachClickHandlers = () => {
        const svgElement = document.getElementById('map-svg');

        if (svgElement) {
            const paths = svgElement.querySelectorAll('path');
            const colors = ['red', 'green', 'yellow'];
            const borderColors = ['blue', 'orange', 'purple'];
        
            paths.forEach((path, index) => {
                path.setAttribute('current', "30%");
                path.setAttribute('prediction', '40%');
                path.addEventListener('mouseenter', (e) => {
                                mouseEntered(e, { name: path.getAttribute('name'), current: path.getAttribute('current'), prediction: path.getAttribute('prediction') });
                            }); 
                path.addEventListener('mouseout', (e) => {
                                mouseGone(e);
                            });
                // Randomly select a color from the array for fill
                const randomFillColor = colors[Math.floor(Math.random() * colors.length)];
                path.style.fill = randomFillColor;
        
                // Randomly select a color from the array for border
                const randomBorderColor = borderColors[Math.floor(Math.random() * borderColors.length)];
                path.style.stroke = 'black';
                path.style.strokeWidth = '2px'; // You can adjust the border width as needed
            });
        }
        

        // if (svgElement) {
        //     const paths = svgElement.querySelectorAll('path');

        //     paths.forEach((path, index) => {
        //         path.addEventListener('click', () => {
        //             handlePathClick(path.getAttribute('id'));
        //         });
        //         path.addEventListener('mouseenter', (e) => {
        //             mouseEntered(e, { name: path.getAttribute('name') });
        //         });
        //         path.addEventListener('mouseout', (e) => {
        //             mouseGone(e);
        //         });
        //     });
        // }
    };

    return (
            <div className="content">
                <div className="map-wrapper">
                    <div id="toolTip" ref={tooltipRef}></div>
                    <div>
                        <ReactSVG
                            src={MySvg}
                            afterInjection={() => {
                                attachClickHandlers()
                            }
                            }
                            beforeInjection={(svg) => {
                                console.log(svg.children);
                                return svg;
                            }}
                            evalScripts="always"
                            fallback={() => <span>Error loading SVG</span>}
                            loading={() => <span>Loading SVG</span>}
                            renumerateIRIElements={false}
                            wrapper="span"
                            className="svg-container"
                            id="my-svg"
                        />
                    </div>
                </div>
            </div>

    );
}

export default Home;
